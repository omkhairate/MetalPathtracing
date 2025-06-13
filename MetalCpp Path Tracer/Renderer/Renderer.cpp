#include "Renderer.h"

#include <simd/simd.h>
#include "Scene.h"
#include "InputSystem.h"
#include "Camera.h"
#include <cstdio>


using namespace MetalCppPathTracer;

struct UniformsData
{
    simd::float3 cameraPosition;

    simd::float2 screenSize;
    
    simd::float3 viewportU;
    simd::float3 viewportV;
    
    simd::float3 firstPixelPosition;
    
    simd::float3 randomSeed;
    
    uint64_t sphereCount;
    
    uint64_t frameCount = 0;
};

inline uint32_t bitm_random()
{
    static uint32_t current_seed = 92407235;
    const uint32_t state = current_seed * 747796405u + 2891336453u;
    const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state);
    return (current_seed = (word >> 22u) ^ word);
}

inline float randomFloat()
{
    return (float)bitm_random() / (float)std::numeric_limits<uint32_t>::max();
}

bool isSphereInFrustum(const simd::float3& position, float radius) {
    // Vector from camera to sphere center
    simd::float3 toSphere = position - Camera::position;

    // Distance along the forward direction
    float forwardDot = simd::dot(toSphere, Camera::forward);

    // Reject spheres behind camera or too far
    if (forwardDot < -radius || forwardDot > 150.0f + radius) return false;

    // Normalize toSphere for angle calculations
    simd::float3 toSphereNorm = simd::normalize(toSphere);

    // Calculate right and up vectors of the camera
    simd::float3 right = simd::normalize(simd::cross(Camera::forward, Camera::up));
    simd::float3 up = simd::normalize(Camera::up);

    // Convert vertical FOV to radians
    float vFov = Camera::verticalFov * (M_PI / 180.0f);

    // Calculate horizontal FOV from vertical FOV and aspect ratio
    float aspectRatio = Camera::screenSize.x / Camera::screenSize.y;
    float hFov = atan(aspectRatio * tan(vFov * 0.5f));

    // Project direction vector onto camera's right and up axes
    float horizontalAngle = acos(simd::dot(toSphereNorm, Camera::forward)); // angle from forward
    float rightDot = simd::dot(toSphereNorm, right);
    float upDot = simd::dot(toSphereNorm, up);

    // Calculate horizontal and vertical angles relative to forward direction
    float angleH = atan2(rightDot, simd::dot(toSphereNorm, Camera::forward)); // left/right angle
    float angleV = atan2(upDot, simd::dot(toSphereNorm, Camera::forward));    // up/down angle

    // Add margin to radius to avoid clipping at edges
    float margin = radius / forwardDot; // approximate angular margin based on radius and distance

    // Check if sphere is inside horizontal and vertical frustum angles (with margin)
    bool insideHorizontal = fabs(angleH) <= hFov + margin;
    bool insideVertical = fabs(angleV) <= vFov * 0.5f + margin;

    return insideHorizontal && insideVertical;
}



Renderer::Renderer( MTL::Device* pDevice )
: _pDevice(pDevice->retain()), _pScene(new Scene(256))
{
    _pCommandQueue = _pDevice->newCommandQueue();
    
    Camera::reset();
    Camera::screenSize = {1280, 720};
    
    updateVisibleScene();
    buildShaders();
    buildBuffers();
    buildTextures();
    
    recalculateViewport();
}

Renderer::~Renderer()
{
    _pSphereBuffer->release();
    _pPSO->release();
    _pCommandQueue->release();
    _pDevice->release();
    
    _pSphereMaterialBuffer->release();
    _pUniformsBuffer->release();
    
    for (int i = 0; i < 2; i++)
        _accumulationTargets[i]->release();
    
    delete _pScene;
}

void Renderer::buildShaders()
{
    using NS::StringEncoding::UTF8StringEncoding;

    NS::Error* pError = nullptr;
    MTL::Library* pLibrary = _pDevice->newDefaultLibrary();
    
    if ( !pLibrary )
    {
        __builtin_printf( "%s", pError->localizedDescription()->utf8String() );
        assert( false );
    }

    MTL::Function* pVertexFn = pLibrary->newFunction( NS::String::string("vertexMain", UTF8StringEncoding) );
    MTL::Function* pFragFn = pLibrary->newFunction( NS::String::string("fragmentMain", UTF8StringEncoding) );

    MTL::RenderPipelineDescriptor* pDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    pDesc->setVertexFunction(pVertexFn);
    pDesc->setFragmentFunction(pFragFn);
    pDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormat::PixelFormatRGBA16Float);

    _pPSO = _pDevice->newRenderPipelineState( pDesc, &pError );
    if ( !_pPSO )
    {
        __builtin_printf( "%s", pError->localizedDescription()->utf8String() );
        assert( false );
    }

    pVertexFn->release();
    pFragFn->release();
    pDesc->release();
    pLibrary->release();
}

void Renderer::updateVisibleScene()
{
    const int sphereCount = 50;
    std::vector<std::pair<simd::float3, float>> visibleSpheres;
    
    if (_allSpheres.empty()) {
        for (size_t i = 0; i < sphereCount; i++) {
            const float radius = randomFloat() * 3 + 0.5f;
            const simd::float3 position = {
                randomFloat() * 100 - 50,
                radius,
                randomFloat() * 100 - 50
            };
            _allSpheres.push_back({position, radius});
        }
    }
    
    for (auto& [position, radius] : _allSpheres) {
        if (isSphereInFrustum(position, radius)) {
            visibleSpheres.push_back({position, radius});
        }
    }
    
    _pScene->clearEntities();


    for (auto& [position, radius] : visibleSpheres) {
        const simd::float3 albedo = { randomFloat(), randomFloat(), randomFloat() };
        const simd::float3 emissionColor = { randomFloat(), randomFloat(), randomFloat() };
        const float materialType = (randomFloat() < 0.3f) ? -1 : (randomFloat() < 0.8f ? 0 : 1.5f);
        const float emissionPower = 0.0f;

        _pScene->addEntity({position, radius}, {albedo, materialType, emissionColor, emissionPower});
    }

    // Add fixed ground and light-emitting spheres
    _pScene->addEntity({{0, -10000, 0}, 10000}, {{0.99, 0.65, 0.01}, 0, {0}, 0});
    _pScene->addEntity({{0, 100, -100}, 40}, {{0.99, 0.65, 0.01}, 0, {0.7, 0.3, 0.1}, 2});
    _pScene->addEntity({{0, 20, -100}, 10}, {{0.99, 0.65, 0.01}, 0, {0.5, 0.8, 0.5}, 20});

    _pScene->buildBVH();
    printf("BVH node count: %zu\n", _pScene->getBVHNodeCount());

    simd::float4* bvhData = _pScene->createBVHBuffer();
    if (_pBVHBuffer) _pBVHBuffer->release();
    _pBVHBuffer = _pDevice->newBuffer(
        bvhData,
        sizeof(simd::float4) * _pScene->getBVHNodeCount() * 2,
        MTL::ResourceStorageModeManaged
    );
    _pBVHBuffer->didModifyRange(NS::Range::Make(0, _pBVHBuffer->length()));

    buildBuffers(); // Rebuild sphere & material buffers
}



void Renderer::recalculateViewport()
{
    float aspectRatio = Camera::screenSize.x / Camera::screenSize.y;

    float fovRad = Camera::verticalFov * (M_PI / 180.0f);
    float halfHeight = tanf(fovRad * 0.5f);
    float halfWidth = aspectRatio * halfHeight;

    simd::float3 w = simd::normalize(-Camera::forward); // camera backward
    simd::float3 u = simd::normalize(simd::cross(Camera::up, w)); // right vector
    simd::float3 v = simd::cross(w, u); // up vector

    simd::float3 viewportU = u * (2.0f * halfWidth);
    simd::float3 viewportV = v * (2.0f * halfHeight);

    // Flip viewportV to fix upside-down
    viewportV = -viewportV;

    // Adjust firstPixelPosition accordingly (assuming top-left corner)
    simd::float3 firstPixelPosition = Camera::position - w - (viewportU * 0.5f) - (viewportV * 0.5f);

    UniformsData* uData = (UniformsData*)_pUniformsBuffer->contents();
    uData->cameraPosition = Camera::position;
    uData->viewportU = viewportU;
    uData->viewportV = viewportV;
    uData->firstPixelPosition = firstPixelPosition;
    uData->screenSize = Camera::screenSize;

    _pUniformsBuffer->didModifyRange(NS::Range::Make(0, sizeof(UniformsData)));
}



void Renderer::buildBuffers()
{
    // Get the current list of visible spheres (all spheres if no culling)
    const size_t sphereCount = _pScene->getEntityCount();
    if (sphereCount == 0) {
        
        if (_pSphereBuffer) { _pSphereBuffer->release(); _pSphereBuffer = nullptr; }
        if (_pSphereMaterialBuffer) { _pSphereMaterialBuffer->release(); _pSphereMaterialBuffer = nullptr; }
        if (_pUniformsBuffer) { _pUniformsBuffer->release(); _pUniformsBuffer = nullptr; }
        return;
    }

    // Get fresh data arrays sized exactly for sphereCount
    simd::float4* sphereTransforms = _pScene->createTransformsBuffer();
    simd::float4* sphereMaterials = _pScene->createMaterialsBuffer();

    // Defensive checks
    assert(sphereTransforms != nullptr && "Transforms buffer is null");
    assert(sphereMaterials != nullptr && "Materials buffer is null");

    // Calculate data sizes carefully — 2 float4's per sphere for materials
    const size_t spheresDataSize = sphereCount * sizeof(simd::float4);
    const size_t sphereMaterialsDataSize = 2 * sphereCount * sizeof(simd::float4);
    const size_t uniformsDataSize = sizeof(UniformsData);

    // Release old buffers safely
    if (_pSphereBuffer) { _pSphereBuffer->release(); _pSphereBuffer = nullptr; }
    if (_pSphereMaterialBuffer) { _pSphereMaterialBuffer->release(); _pSphereMaterialBuffer = nullptr; }
    if (_pUniformsBuffer) { _pUniformsBuffer->release(); _pUniformsBuffer = nullptr; }

    // Create new buffers with exact size
    _pSphereBuffer = _pDevice->newBuffer(spheresDataSize, MTL::ResourceStorageModeManaged);
    _pSphereMaterialBuffer = _pDevice->newBuffer(sphereMaterialsDataSize, MTL::ResourceStorageModeManaged);
    _pUniformsBuffer = _pDevice->newBuffer(uniformsDataSize, MTL::ResourceStorageModeManaged);

    // Copy data to GPU buffers
    memcpy(_pSphereBuffer->contents(), sphereTransforms, spheresDataSize);
    memcpy(_pSphereMaterialBuffer->contents(), sphereMaterials, sphereMaterialsDataSize);

    // Tell Metal buffers what range we modified
    _pSphereBuffer->didModifyRange(NS::Range::Make(0, spheresDataSize));
    _pSphereMaterialBuffer->didModifyRange(NS::Range::Make(0, sphereMaterialsDataSize));
    _pUniformsBuffer->didModifyRange(NS::Range::Make(0, uniformsDataSize));
}



void Renderer::buildTextures()
{
    MTL::TextureDescriptor *textureDescriptor = MTL::TextureDescriptor::alloc()->init();

    textureDescriptor->setPixelFormat(MTL::PixelFormat::PixelFormatRGBA32Float);
    textureDescriptor->setTextureType(MTL::TextureType::TextureType2D);
    textureDescriptor->setWidth(Camera::screenSize.x);
    textureDescriptor->setHeight(Camera::screenSize.y);
    textureDescriptor->setStorageMode(MTL::StorageMode::StorageModePrivate);
    textureDescriptor->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);

    for (uint i = 0; i < 2; i++)
        _accumulationTargets[i] = _pDevice->newTexture(textureDescriptor);
}

bool Renderer::updateCamera()
{
    const bool cameraUpdated = Camera::transformWithInputs();
    
    if(cameraUpdated) recalculateViewport();
    
    return cameraUpdated;
}


void Renderer::updateUniforms()
{
    UniformsData &u = *((UniformsData*)_pUniformsBuffer->contents());
    

    const bool cameraChanged = updateCamera();

    if (cameraChanged)
    {
        u.frameCount = 0;  // reset accumulation
        // clear accumulation buffer if you have one
        u.randomSeed = {randomFloat(), randomFloat(), randomFloat()};
    }
    else
    {
        u.frameCount += 1;
    }

    
    u.sphereCount = _pScene->getEntityCount();

    _pUniformsBuffer->didModifyRange(NS::Range::Make(0, _pUniformsBuffer->length()));
}


void Renderer::draw( MTK::View* pView )
{
    static int frameCounter = 0;
    frameCounter++;

    if (frameCounter % 30 == 0)
        updateVisibleScene();

    {
        updateUniforms();
        MTL::Texture *tmp = _accumulationTargets[0];
        _accumulationTargets[0] = _accumulationTargets[1];
        _accumulationTargets[1] = tmp;
    }

    {
        NS::AutoreleasePool *pPool = NS::AutoreleasePool::alloc()->init();

        MTL::CommandBuffer *pCmd = _pCommandQueue->commandBuffer();
        MTL::RenderPassDescriptor *pRpd = pView->currentRenderPassDescriptor();
        MTL::RenderCommandEncoder *pEnc = pCmd->renderCommandEncoder(pRpd);

        pEnc->setRenderPipelineState(_pPSO);

        pEnc->setFragmentBuffer(_pSphereBuffer, 0, 0);
        pEnc->setFragmentBuffer(_pSphereMaterialBuffer, 0, 1);
        pEnc->setFragmentBuffer(_pUniformsBuffer, 0, 2);

        pEnc->setFragmentTexture(_accumulationTargets[0], 0);
        pEnc->setFragmentTexture(_accumulationTargets[1], 1);
        pEnc->drawPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangle, NS::UInteger(0),
                             NS::UInteger(6));

        pEnc->endEncoding();
        pCmd->presentDrawable(pView->currentDrawable());
        pCmd->commit();

        pPool->release();
    }
}


void Renderer::drawableSizeWillChange(MTK::View *pView, CGSize size)
{
    for (uint i = 0; i < 2; i++)
        _accumulationTargets[i]->release();
    
    Camera::screenSize = {(float)size.width, (float)size.height};
    
    buildTextures();
    
    recalculateViewport();
}
