class Texture:
    def __init__(self, texture_entry, gltf_loader):
        self._index = texture_entry['index'] if ('index' in texture_entry) else 0
        self._texcoord = texture_entry['texCoord'] if ('texCoord' in texture_entry) else 0

class NormalTexture(Texture):
    def __init__(self, normal_texture_entry):
        super(normal_texture_entry)
        self._scale = normal_texture_entry['scale'] if ('scale' in normal_texture_entry) else 1.0

class OcclusionTexture(Texture):
    def __init__(self, occlusion_texture_entry):
        super(occlusion_texture_entry)
        self._strength = occlusion_texture_entry['strength'] if ('strength' in occlusion_texture_entry) else 1.0

class PbrMetallicRoughness:
    def __init__(self, pbr_metallic_roughness_entry, gltf_loader):
        self._base_color_factor = pbr_metallic_roughness_entry['baseColorFactor'] if ('baseColorFactor' in pbr_metallic_roughness_entry) else [1.0,1.0,1.0]
        self._metallic_factor = pbr_metallic_roughness_entry['metallicFactor'] if ('metallicFactor' in pbr_metallic_roughness_entry) else 1.0
        self._roughness_factor = pbr_metallic_roughness_entry['roughnessFactor'] if ('roughnessFactor' in pbr_metallic_roughness_entry) else 1.0
        self._base_color_texture = Texture(pbr_metallic_roughness_entry['baseColorTexture'], gltf_loader) if ('baseColorTexture' in pbr_metallic_roughness_entry) else None
        self._metallic_roughness_texture = Texture(pbr_metallic_roughness_entry['metallicRoughnessTexture'], gltf_loader) if ('metallicRoughnessTexture' in pbr_metallic_roughness_entry) else None

class PbrSpecularGlossiness:
    def __init__(self, pbr_specular_glossiness_entry, gltf_loader):
        self._diffuse_factor = pbr_specular_glossiness_entry['diffuseFactor'] if ('diffuseFactor' in pbr_specular_glossiness_entry) else [1.0,1.0,1.0,1.0]
        self.diffuse_texture = Texture(pbr_specular_glossiness_entry['diffuseTexture'], gltf_loader) if ('diffuseTexture' in pbr_specular_glossiness_entry) else None
        self._specular_factor = pbr_specular_glossiness_entry['specularFactor'] if ('specularFactor' in pbr_specular_glossiness_entry) else [1.0,1.0,1.0]
        self._glossiness_factor = pbr_specular_glossiness_entry['glossinessFactor'] if ('glossinessFactor' in pbr_specular_glossiness_entry) else 1.0
        self._specular_glossiness_texture = Texture(pbr_specular_glossiness_entry['specularGlossinessTexture'], gltf_loader) if ('specularGlossinessTexture' in pbr_specular_glossiness_entry) else None


class Material:
    def __init__(self, material_entry, material_index, gltf_loader):
        self._name = material_entry['name'] if ('name' in material_entry) else 'material_{}'.format(material_index)
        self._index = material_index
        self._double_sided = material_entry['doubleSided'] if ('doubleSided' in material_entry) else False

        self._pbr_metallic_roughness = PbrMetallicRoughness(material_entry['pbrMetallicRoughness'], gltf_loader) if ('pbrMetallicRoughness' in material_entry) else None

        self._normal_texture = Texture(material_entry['normalTexture'], gltf_loader) if ('normalTexture' in material_entry) else None
        self._emissiveFactor = material_entry['emissiveFactor'] if ('emissiveFactor' in material_entry) else [0,0,0]

        self._extensions = {}
        if 'extensions' in material_entry and 'KHR_materials_pbrSpecularGlossiness' in material_entry['extensions']:
            self._extensions['KHR_materials_pbrSpecularGlossiness'] = PbrSpecularGlossiness(material_entry['extensions']['KHR_materials_pbrSpecularGlossiness'], gltf_loader)

    def is_double_sided(self):
        return self._double_sided

    def get_index(self):
        return self._index