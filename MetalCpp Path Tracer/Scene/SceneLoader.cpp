#include "SceneLoader.h"
#include "Transform.h"
#include "Material.h"
#include <tinyxml2.h>
#include <simd/simd.h>
#include <cstdio>

using namespace tinyxml2;

namespace MetalCppPathTracer {

// Utility to parse a vec3 string
static simd::float3 parseVec3(const char* str) {
    float x=0,y=0,z=0;
    sscanf(str, "%f,%f,%f", &x, &y, &z);
    return simd::make_float3(x,y,z);
}

void SceneLoader::LoadSceneFromXML(const std::string& path, Scene* scene) {
    XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != XML_SUCCESS) {
        printf("Failed to load scene XML: %s\n", path.c_str());
        return;
    }

    scene->clearEntities();

    XMLElement* root = doc.FirstChildElement("Scene");
    if (!root) {
        printf("Scene XML has no <Scene> root.\n");
        return;
    }

    for (XMLElement* e = root->FirstChildElement("Sphere"); e; e = e->NextSiblingElement("Sphere")) {
        Transform t;
        t.position = parseVec3(e->Attribute("position"));
        t.scale = e->FloatAttribute("radius");

        Material m;
        m.albedo = parseVec3(e->Attribute("albedo"));
        m.emissionColor = parseVec3(e->Attribute("emission"));
        m.materialType = e->FloatAttribute("materialType", 0);
        m.emissionPower = e->FloatAttribute("emissionPower", 0);

        scene->addEntity(t, m);
    }
}

}
