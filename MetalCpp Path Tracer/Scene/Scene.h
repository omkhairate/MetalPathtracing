#ifndef SCENE_H
#define SCENE_H

#include "Transform.h"
#include "Material.h"
#include <vector>
#include <limits>
#include <algorithm>
#include <simd/simd.h>

namespace MetalCppPathTracer {

struct BVHNode {
    simd::float3 boundsMin;
    simd::float3 boundsMax;
    int leftFirst; // either child index or first primitive index
    int count;     // 0 for internal node, >0 for leaf node
};

class Scene
{
 public:
    Scene(size_t capacity) : materials(new Material[capacity]), transforms(new Transform[capacity]) {}
    
    inline size_t addEntity(const Transform &transform, const Material &mat)
    {
        materials[entityCount] = mat;
        transforms[entityCount] = transform;
        return entityCount++;
    }
    
    inline size_t addEntity(const Transform &&transform, const Material &&mat)
    {
        materials[entityCount] = mat;
        transforms[entityCount] = transform;
        return entityCount++;
    }
    
    inline simd::float4 *createMaterialsBuffer()
    {
        simd::float4 *buffer = new simd::float4[2 * entityCount];
        
        for(size_t i = 0; i < entityCount; i++)
        {
            const Material &mat = materials[i];
            
            buffer[2*i] = simd::make_float4(mat.albedo, mat.materialType);
            buffer[2*i + 1] = simd::make_float4(mat.emissionColor, mat.emissionPower);
        }
        
        return buffer;
    }
    
    inline simd::float4 *createTransformsBuffer()
    {
        simd::float4 *buffer = new simd::float4[entityCount];
        
        for(size_t i = 0; i < entityCount; i++)
        {
            const Transform &transform = transforms[i];
            
            buffer[i] = simd::make_float4(transform.position, transform.scale);
        }
        
        return buffer;
    }
    
    inline size_t getEntityCount()
    {
        return entityCount;
    }
    
    void buildBVH() {
        primitiveIndices.resize(entityCount);
        for (size_t i = 0; i < entityCount; ++i)
            primitiveIndices[i] = i;

        bvhNodes.clear();
        buildBVHRecursive(0, entityCount);
    }

    simd::float4 *createBVHBuffer() {
        simd::float4 *buffer = new simd::float4[bvhNodes.size() * 2];
        for (size_t i = 0; i < bvhNodes.size(); ++i) {
            const BVHNode &node = bvhNodes[i];

            buffer[2 * i]     = simd::make_float4(node.boundsMin, *(float *)&node.leftFirst);
            buffer[2 * i + 1] = simd::make_float4(node.boundsMax, *(float *)&node.count);
        }

        return buffer;
    }

    inline size_t getBVHNodeCount() const {
        return bvhNodes.size();
    }
    
    
    inline void clearEntities() {
        entityCount = 0;
        bvhNodes.clear();
        primitiveIndices.clear();
    }
    

    

    
 private:
    size_t entityCount = 0;
    size_t capacity;

    Material *materials;
    Transform *transforms;

    std::vector<BVHNode> bvhNodes;
    std::vector<size_t> primitiveIndices;

    int buildBVHRecursive(size_t start, size_t end) {
        BVHNode node;

        simd::float3 boundsMin( std::numeric_limits<float>::max());
        simd::float3 boundsMax(-std::numeric_limits<float>::max());

        for (size_t i = start; i < end; ++i) {
            const Transform &t = transforms[primitiveIndices[i]];
            simd::float3 p = t.position;
            float s = t.scale;

            boundsMin = simd::min(boundsMin, p - s);
            boundsMax = simd::max(boundsMax, p + s);
        }

        node.boundsMin = boundsMin;
        node.boundsMax = boundsMax;
        node.leftFirst = static_cast<int>(start);
        node.count = static_cast<int>(end - start);

        int nodeIndex = static_cast<int>(bvhNodes.size());
        bvhNodes.push_back(node);

        if (node.count <= 2) return nodeIndex;

        simd::float3 extent = boundsMax - boundsMin;
        int axis = extent.x > extent.y ? (extent.x > extent.z ? 0 : 2) : (extent.y > extent.z ? 1 : 2);
        float splitPos = 0.5f * (boundsMin[axis] + boundsMax[axis]);

        auto midIter = std::partition(
            primitiveIndices.begin() + start,
            primitiveIndices.begin() + end,
            [&](size_t i) {
                return transforms[i].position[axis] < splitPos;
            });

        size_t mid = std::distance(primitiveIndices.begin(), midIter);
        if (mid == start || mid == end)
            mid = start + (end - start) / 2;

        int leftChild = buildBVHRecursive(start, mid);
        int rightChild = buildBVHRecursive(mid, end);

        bvhNodes[nodeIndex].leftFirst = leftChild;
        bvhNodes[nodeIndex].count = 0; // Internal node

        return nodeIndex;
    }
};


}

#endif
