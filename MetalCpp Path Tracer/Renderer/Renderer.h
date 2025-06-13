#ifndef RENDERER_H
#define RENDERER_H


#include <Metal/Metal.hpp>
#include <MetalKit/MetalKit.hpp>
#include <simd/simd.h>

#include "Scene.h"

namespace MetalCppPathTracer
{

class Renderer
{
 public:
    Renderer(MTL::Device *pDevice);
    ~Renderer();
    void updateVisibleScene();
    void buildShaders();
    void buildBuffers();
    void buildTextures();
    
    void recalculateViewport();
    bool updateCamera();
    
    void updateUniforms();
    
    void draw(MTK::View *pView);
    void drawableSizeWillChange(MTK::View *pView, CGSize size);
    std::vector<std::pair<simd::float3, float>> _allSpheres;

    
    struct Chunk {
        std::vector<std::pair<simd::float4, simd::float4>> spheres; // (transform, material)
        simd::int3 chunkCoords;
    };

    
 private:
    MTL::Device *_pDevice;
    MTL::CommandQueue *_pCommandQueue;
    MTL::RenderPipelineState *_pPSO;
    
    MTL::Buffer *_pSphereBuffer;
    MTL::Buffer *_pSphereMaterialBuffer;
    MTL::Buffer *_pUniformsBuffer;
    MTL::Texture *_accumulationTargets[2];
    MTL::Buffer *_pBVHBuffer;

    
    Scene *_pScene;
};

};

#endif  //  RENDERER_H
