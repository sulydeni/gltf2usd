import argparse
import base64
import collections
import json
import logging
import ntpath
import numpy
import os
import re
import shutil

from io import BytesIO

from gltf2loader import GLTF2Loader, PrimitiveMode, TextureWrap, MinFilter, MagFilter

from PIL import Image

from pxr import Usd, UsdGeom, Sdf, UsdShade, Gf, UsdSkel, Vt, Ar, UsdUtils

from gltf2loader import GLTF2Loader, PrimitiveMode, TextureWrap, MinFilter, MagFilter
from gltf2usdUtils import GLTF2USDUtils
from usd_material import USDMaterial

class GLTF2USD:
    """
    Class for converting glTF 2.0 models to Pixar's USD format.  Currently openly supports .gltf files
    with non-embedded data and exports to .usda .
    """

    TEXTURE_SAMPLER_WRAP = {
        TextureWrap.CLAMP_TO_EDGE : 'clamp',
        TextureWrap.MIRRORED_REPEAT : 'mirror',
        TextureWrap.REPEAT: 'repeat',
    }

    def __init__(self, gltf_file, usd_file, fps, scale, verbose=False):
        """Initializes the glTF to USD converter

        Arguments:
            gltf_file {str} -- path to the glTF file
            usd_file {str} -- path to store the generated usda file
            verbose {boolean} -- specifies if the output should be verbose from this tool
        """
        self.logger = logging.getLogger('gltf2usd')
        self.logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)

        self.fps = fps
        self.gltf_loader = GLTF2Loader(gltf_file)
        self.verbose = verbose
        self.scale = scale

        self.output_dir = os.path.dirname(os.path.abspath(usd_file))
        filename = '%s.%s' % (os.path.splitext(usd_file)[0], 'usda')

        #if usdz file is desired, change to usdc file
        if usd_file.endswith('usdz'):
            usd_file = usd_file[:-1] + 'c'
            self.logger.info("converted usd file extension from .usdz to .usdc: {}".format(usd_file))
        self.stage = Usd.Stage.CreateNew(usd_file)
        self.gltf_usd_nodemap = {}
        self.gltf_usdskel_nodemap = {}
        self._usd_mesh_skin_map = {}
        self._joint_hierarchy_name_map = {}

        self.convert()


    def convert_nodes_to_xform(self):
        """
        Converts the glTF nodes to USD Xforms.  The models get a parent Xform that scales the geometry by 100
        to convert from meters (glTF) to centimeters (USD).
        """
        parent_transform = UsdGeom.Xform.Define(self.stage, '/root')
        parent_transform.AddScaleOp().Set((self.scale, self.scale, self.scale))

        main_scene = self.gltf_loader.get_main_scene()

        nodes = main_scene.get_nodes()
        root_nodes = [node for node in nodes if node.get_parent() == None]
        for node in root_nodes:
            self._convert_node_to_xform(node, parent_transform)


    def _convert_node_to_xform(self, node, usd_xform):
        """Converts a glTF node to a USD transform node.

        Arguments:
            node {dict} -- glTF node
            node_index {int} -- glTF node index
            xform_name {str} -- USD xform name
        """        
        xformPrim = UsdGeom.Xform.Define(self.stage, '{0}/{1}'.format(usd_xform.GetPath(), GLTF2USDUtils.convert_to_usd_friendly_node_name(node.get_name())))
        xformPrim.AddTransformOp().Set(self._compute_rest_matrix(node))
        mesh = node.get_mesh()
        if mesh != None:
            usd_mesh = self._convert_mesh_to_xform(mesh, xformPrim, node)

        self._convert_animation_to_usd(node, xformPrim)
        
        children = node.get_children()

        for child in children:
            self._convert_node_to_xform(child, xformPrim)


    def _create_usd_skeleton(self, gltf_skin, usd_xform, usd_joint_names):
        """Creates a USD skeleton from a glTF skin
        
        Arguments:
            gltf_skin {Skin} -- gltf skin
            usd_xform {Xform} -- USD Xform
        
        Returns:
            Skeleton -- USD skeleton
        """

        # create skeleton  
        root_joint = gltf_skin.get_root_joint()
        root_joint_name = GLTF2USDUtils.convert_to_usd_friendly_node_name(root_joint.get_name())

        skeleton = UsdSkel.Skeleton.Define(self.stage, '{0}/{1}'.format(usd_xform.GetPath(), root_joint_name))

        gltf_bind_transforms = [Gf.Matrix4d(*xform).GetInverse() for xform in gltf_skin.get_inverse_bind_matrices()]
        gltf_rest_transforms = [GLTF2USDUtils.compute_usd_transform_matrix_from_gltf_node(joint) for joint in gltf_skin.get_joints()]
        skeleton.CreateJointsAttr().Set(usd_joint_names)
        skeleton.CreateBindTransformsAttr(gltf_bind_transforms)
        skeleton.CreateRestTransformsAttr(gltf_rest_transforms)

        return skeleton
  
    def _create_usd_skeleton_animation(self, gltf_skin, usd_skeleton, joint_names):
        #get the animation data per joint
        skelAnim = UsdSkel.Animation.Define(self.stage, '{0}/{1}'.format(usd_skeleton.GetPath(), 'anim'))

        usd_skel_root_path = usd_skeleton.GetPath().GetParentPath()
        usd_skel_root = self.stage.GetPrimAtPath(usd_skel_root_path)


        skelAnim.CreateJointsAttr().Set(joint_names)

        gltf_animation = self.gltf_loader.get_animations()[0]
        min_sample = 999
        max_sample = -999
        for sampler in gltf_animation.get_samplers():
            input_data = sampler.get_input_data()
            min_sample = min(min_sample, input_data[0])
            max_sample = max(max_sample, input_data[-1])

        rotate_attr = skelAnim.CreateRotationsAttr()
        for input_key in numpy.arange(min_sample, max_sample, 1./self.fps):
            entries = []
            for joint in gltf_skin.get_joints():
                anim = self._get_anim_data_for_joint_and_path(gltf_animation, joint, 'rotation', input_key)
                entries.append(anim)

            if len(gltf_skin.get_joints()) != len(entries):
                raise Exception('up oh!')

            rotate_attr.Set(Vt.QuatfArray(entries), Usd.TimeCode(input_key * self.fps))

        translate_attr = skelAnim.CreateTranslationsAttr()
        for input_key in numpy.arange(min_sample, max_sample, 1./self.fps):
            entries = []
            for joint in gltf_skin.get_joints():
                anim = self._get_anim_data_for_joint_and_path(gltf_animation, joint, 'translation', input_key)
                entries.append(anim)

            if len(gltf_skin.get_joints()) != len(entries):
                raise Exception('up oh!')

            translate_attr.Set(entries, Usd.TimeCode(input_key * self.fps))

        scale_attr = skelAnim.CreateScalesAttr()
        for input_key in numpy.arange(min_sample, max_sample, 1./self.fps):
            entries = []
            for joint in gltf_skin.get_joints():
                anim = self._get_anim_data_for_joint_and_path(gltf_animation, joint, 'scale', input_key)
                entries.append(anim)

            if len(gltf_skin.get_joints()) != len(entries):
                raise Exception('up oh!')

            scale_attr.Set(entries, Usd.TimeCode(input_key * self.fps))

        return skelAnim

    def _get_anim_data_for_joint_and_path(self, gltf_animation, gltf_joint, path, time_sample):
        anim_channel = gltf_animation.get_animation_channel_for_node_and_path(gltf_joint, path)
        if not anim_channel:
            if path == 'translation':
                return gltf_joint.get_translation()
            elif path == 'rotation':
                gltf_rotation = gltf_joint.get_rotation()
                usd_rotation = Gf.Quatf(gltf_rotation[3], gltf_rotation[0], gltf_rotation[1], gltf_rotation[2])
                return usd_rotation
            elif path == 'scale':
                return gltf_joint.get_scale()

        else:
            if path == 'rotation':
                rotation = anim_channel.get_sampler().get_interpolated_output_data(time_sample)
                return rotation
            elif path == 'scale' or path =='translation':
                return anim_channel.get_sampler().get_interpolated_output_data(time_sample)
            else:
                raise Exception('unsupported animation type: {}'.format(path))
                
            
    def _get_usd_joint_hierarchy_name(self, gltf_joint, root_joint):
        if gltf_joint in self._joint_hierarchy_name_map:
            return GLTF2USDUtils.convert_to_usd_friendly_node_name(self._joint_hierarchy_name_map[gltf_joint])
        
        joint = gltf_joint
        joint_name_stack = [GLTF2USDUtils.convert_to_usd_friendly_node_name(joint.get_name())]

        while joint.get_parent() != None and joint != root_joint:
            joint = joint.get_parent()
            joint_name_stack.append(GLTF2USDUtils.convert_to_usd_friendly_node_name(joint.get_name()))

        joint_name = ''
        while len(joint_name_stack) > 0:
            if joint_name:
                joint_name = '{0}/{1}'.format(joint_name, joint_name_stack.pop())
            else:
                joint_name = joint_name_stack.pop()

        self._joint_hierarchy_name_map[gltf_joint] = joint_name
        return GLTF2USDUtils.convert_to_usd_friendly_node_name(joint_name)


    def _convert_mesh_to_xform(self, gltf_mesh, usd_node, gltf_node):
        """
        Converts a glTF mesh to a USD Xform.
        Each primitive becomes a submesh of the Xform.

        Arguments:
            mesh {dict} -- glTF mesh
            parent_path {str} -- xform path
            node_index {int} -- glTF node index
        """
        #for each mesh primitive, create a USD mesh
        for primitive in gltf_mesh.get_primitives():
            self._convert_primitive_to_mesh(primitive, usd_node, gltf_node, gltf_mesh)

    def _convert_primitive_to_mesh(self, gltf_primitive, usd_node, gltf_node, gltf_mesh):
        """
        Converts a glTF mesh primitive to a USD mesh

        Arguments:
            name {str} -- name of the primitive
            primitive {dict} -- glTF primitive
            usd_parent_node {str} -- USD parent xform
            node_index {int} -- glTF node index
            double_sided {bool} -- specifies if the primitive is double sided
        """
        parent_node = usd_node
        parent_path = parent_node.GetPath()
        attributes = gltf_primitive.get_attributes()
        skel_root = None
        targets = gltf_primitive.get_morph_targets()
        if 'JOINTS_0' in attributes or len(targets) > 0:
            skeleton_path = '{0}/{1}'.format(usd_node.GetPath(),  'skeleton_root')
            skel_root = UsdSkel.Root.Define(self.stage, skeleton_path)
            parent_node = skel_root
            parent_path = parent_node.GetPath()
        mesh = UsdGeom.Mesh.Define(self.stage, '{0}/{1}'.format(parent_node.GetPath(), GLTF2USDUtils.convert_to_usd_friendly_node_name(gltf_primitive.get_name())))
        mesh.CreateSubdivisionSchemeAttr().Set('none')

        material = gltf_primitive.get_material()
        if material != None:
            if material.is_double_sided():
                mesh.CreateDoubleSidedAttr().Set(True)

            usd_material = self.usd_materials[material.get_index()]
            UsdShade.MaterialBindingAPI(mesh).Bind(usd_material.get_usd_material())

        for attribute_name in attributes:
            attribute = attributes[attribute_name]
            if attribute_name == 'POSITION':
                override_prim = self.stage.OverridePrim(mesh.GetPath())
                override_prim.CreateAttribute('extent', Sdf.ValueTypeNames.Float3Array).Set([attribute.get_min_value(), attribute.get_max_value()])
                mesh.CreatePointsAttr(attribute.get_data())

            if attribute_name == 'NORMAL':
                mesh.CreateNormalsAttr(attribute.get_data())

            if attribute_name == 'COLOR_0':
                prim_var = UsdGeom.PrimvarsAPI(mesh)
                colors = prim_var.CreatePrimvar('displayColor', Sdf.ValueTypeNames.Color3f, 'vertex').Set(attribute.get_data())

            if attribute_name == 'TEXCOORD_0':
                data = attribute.get_data()
                invert_uvs = []
                for uv in data:
                    new_uv = (uv[0], 1 - uv[1])
                    invert_uvs.append(new_uv)
                prim_var = UsdGeom.PrimvarsAPI(mesh)
                uv = prim_var.CreatePrimvar('primvars:st0', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex')
                uv.Set(invert_uvs)
            if attribute_name == 'JOINTS_0':
                self._convert_skin_to_usd(gltf_node, gltf_primitive, parent_node, mesh)
        
        weights = gltf_mesh.get_weights()
        if targets:
            skinBinding = UsdSkel.BindingAPI.Apply(mesh.GetPrim())

            skeleton = UsdSkel.Skeleton.Define(self.stage, '{0}/skel'.format(parent_path))

            # Create an animation for this mesh to hold the blendshapes
            skelAnim = UsdSkel.Animation.Define(self.stage, '{0}/skel/anim'.format(parent_path))

            # link the skeleton animation to skelAnim
            skinBinding.CreateAnimationSourceRel().AddTarget(skelAnim.GetPath())

            # link the skeleton to skeleton
            skinBinding.CreateSkeletonRel().AddTarget(skeleton.GetPath())

            # Set blendshape names on the animation
            names = []
            for i, _ in enumerate(gltf_mesh.get_weights()):
                targets[i].get_name()
                blend_shape_name = targets[i].get_name()
                names.append(blend_shape_name)

            skelAnim.CreateBlendShapesAttr().Set(names)
            skinBinding.CreateBlendShapesAttr(names)

            # Set the starting weights of each blendshape to the weights defined in the glTF primitive
            skelAnim.CreateBlendShapeWeightsAttr().Set(weights)
            blend_shape_targets = skinBinding.CreateBlendShapeTargetsRel()

            # Define offsets for each blendshape, and add them as skel:blendShapes and skel:blendShapeTargets
            for i, name in enumerate(names):
                offsets = targets[i].get_attributes()['POSITION']
                blend_shape_name = '{0}/{1}'.format(mesh.GetPath(), name)

                # define blendshapes in the mesh
                blend_shape = UsdSkel.BlendShape.Define(self.stage, blend_shape_name)
                blend_shape.CreateOffsetsAttr(offsets)
                blend_shape_targets.AddTarget(name)

        indices = gltf_primitive.get_indices()
        num_faces = len(indices)/3
        face_count = [3] * num_faces
        mesh.CreateFaceVertexCountsAttr(face_count)
        mesh.CreateFaceVertexIndicesAttr(indices)

    def _get_texture__wrap_modes(self, texture):
        """Get the USD texture wrap modes from a glTF texture

        Arguments:
            texture {dict} -- glTF texture

        Returns:
            dict -- dictionary mapping wrapS and wrapT to
            a USD texture sampler mode
        """

        texture_data = {'wrapS': 'repeat', 'wrapT': 'repeat'}
        if 'sampler' in texture:
            sampler = self.gltf_loader.json_data['samplers'][texture['sampler']]

            if 'wrapS' in sampler:
                texture_data['wrapS'] = GLTF2USD.TEXTURE_SAMPLER_WRAP[TextureWrap(sampler['wrapS'])]

            if 'wrapT' in sampler:
                texture_data['wrapT'] = GLTF2USD.TEXTURE_SAMPLER_WRAP[TextureWrap(sampler['wrapT'])]

        return texture_data

    def _convert_images_to_usd(self):
        """
        Converts the glTF images to USD images
        """

        if 'images' in self.gltf_loader.json_data:
            self.images = []
            for i, image in enumerate(self.gltf_loader.json_data['images']):
                image_name = ''

                # save data-uri textures
                if image['uri'].startswith('data:image'):
                    uri_data = image['uri'].split(',')[1]
                    img = Image.open(BytesIO(base64.b64decode(uri_data)))

                    # NOTE: image might not have a name
                    image_name = image['name'] if 'name' in image else 'image{}.{}'.format(i, img.format.lower())
                    image_path = os.path.join(self.gltf_loader.root_dir, image_name)
                    img.save(image_path)

                # otherwise just copy the texture over
                else:
                    image_path = os.path.join(self.gltf_loader.root_dir, image['uri'])
                    image_name = os.path.join(self.output_dir, ntpath.basename(image_path))

                    if self.gltf_loader.root_dir is not self.output_dir:
                        shutil.copyfile(image_path, image_name)

                self.images.append(ntpath.basename(image_name))

    def _convert_materials_to_preview_surface(self):
        """
        Converts the glTF materials to preview surfaces
        """

        if 'materials' in self.gltf_loader.json_data:
            self.usd_materials = []
            material_path_root = '/Materials'
            scope = UsdGeom.Scope.Define(self.stage, material_path_root)

            for i, material in enumerate(self.gltf_loader.json_data['materials']):
                name = 'pbrmaterial_{}'.format(i)
                material_path = Sdf.Path('{0}/{1}'.format(material_path_root, name))
                usd_material = UsdShade.Material.Define(self.stage, material_path)
                self.usd_materials.append(usd_material)

                usd_material_surface_output = usd_material.CreateOutput("surface", Sdf.ValueTypeNames.Token)
                usd_material_displacement_output = usd_material.CreateOutput("displacement", Sdf.ValueTypeNames.Token)
                pbr_mat = UsdShade.Shader.Define(self.stage, material_path.AppendChild('pbrMat1'))
                pbr_mat.CreateIdAttr("UsdPreviewSurface")
                specular_workflow = pbr_mat.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Bool)
                specular_workflow.Set(False)
                pbr_mat_surface_output = pbr_mat.CreateOutput("surface", Sdf.ValueTypeNames.Token)
                pbr_mat_displacement_output = pbr_mat.CreateOutput("displacement", Sdf.ValueTypeNames.Token)
                usd_material_surface_output.ConnectToSource(pbr_mat_surface_output)
                usd_material_displacement_output.ConnectToSource(pbr_mat_displacement_output)

                #define uv primvar0
                primvar_st0 = UsdShade.Shader.Define(self.stage, material_path.AppendChild('primvar_st0'))
                primvar_st0.CreateIdAttr('UsdPrimvarReader_float2')
                fallback_st0 = primvar_st0.CreateInput('fallback', Sdf.ValueTypeNames.Float2)
                fallback_st0.Set((0,0))
                primvar_st0_varname = primvar_st0.CreateInput('varname', Sdf.ValueTypeNames.Token)
                primvar_st0_varname.Set('st0')
                primvar_st0_output = primvar_st0.CreateOutput('result', Sdf.ValueTypeNames.Float2)

                #define uv primvar1
                primvar_st1 = UsdShade.Shader.Define(self.stage, material_path.AppendChild('primvar_st1'))
                primvar_st1.CreateIdAttr('UsdPrimvarReader_float2')
                fallback_st1 = primvar_st1.CreateInput('fallback', Sdf.ValueTypeNames.Float2)
                fallback_st1.Set((0,0))
                primvar_st1_varname = primvar_st1.CreateInput('varname', Sdf.ValueTypeNames.Token)
                primvar_st1_varname.Set('st1')
                primvar_st1_output = primvar_st1.CreateOutput('result', Sdf.ValueTypeNames.Float2)

                pbr_metallic_roughness = None
                pbr_specular_glossiness = None

                if 'pbrMetallicRoughness' in material:
                    pbr_metallic_roughness = material['pbrMetallicRoughness']
                    if 'baseColorFactor' in pbr_metallic_roughness:
                        diffuse_color = pbr_mat.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
                        base_color_factor = pbr_metallic_roughness['baseColorFactor']
                        diffuse_color.Set((base_color_factor[0],base_color_factor[1],base_color_factor[2]))
                        opacity = pbr_mat.CreateInput("opacity", Sdf.ValueTypeNames.Float)
                        opacity.Set(base_color_factor[3])
                    if 'metallicFactor' in pbr_metallic_roughness:
                        metallic_factor = pbr_metallic_roughness['metallicFactor']
                        metallic = pbr_mat.CreateInput('metallic', Sdf.ValueTypeNames.Float)
                        metallic.Set(pbr_metallic_roughness['metallicFactor'])
                elif 'extensions' in material and 'KHR_materials_pbrSpecularGlossiness' in material['extensions']:
                    specular_workflow.Set(True)
                    pbr_specular_glossiness = material['extensions']['KHR_materials_pbrSpecularGlossiness']

                    specular_factor = pbr_specular_glossiness['specularFactor'] if ('specularFactor' in pbr_specular_glossiness) else (1,1,1)
                    pbr_mat.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((specular_factor[0], specular_factor[1], specular_factor[2]))
                    diffuse_factor = pbr_specular_glossiness['diffuseFactor'] if ('diffuseFactor' in pbr_specular_glossiness) else (1,1,1,1)
                    diffuse_color = pbr_mat.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
                    diffuse_color.Set((diffuse_factor[0], diffuse_factor[1], diffuse_factor[2]))
                    opacity = pbr_mat.CreateInput("opacity", Sdf.ValueTypeNames.Float)
                    opacity.Set(diffuse_factor[3])

                if 'occlusionTexture' in material:
                    occlusion_texture = material['occlusionTexture']
                    scale_factor = occlusion_texture['strength'] if 'strength' in occlusion_texture else 1
                    fallback_occlusion_value = scale_factor
                    scale_factor = (scale_factor, scale_factor, scale_factor, 1)
                    occlusion_components = {
                        'r':
                        {'sdf_type' : Sdf.ValueTypeNames.Float, 'name': 'occlusion'}
                    }

                    self._convert_texture_to_usd(
                        pbr_mat=pbr_mat,
                        gltf_texture=occlusion_texture,
                        gltf_texture_name= 'occlusionTexture',
                        color_components= occlusion_components,
                        scale_factor=scale_factor,
                        fallback_factor=fallback_occlusion_value,
                        material_path=material_path,
                        fallback_type=Sdf.ValueTypeNames.Float,
                        primvar_st0_output=primvar_st0_output,
                        primvar_st1_output=primvar_st1_output
                    )


                if 'normalTexture' in material:
                    normal_texture = material['normalTexture']
                    scale_factor = normal_texture['scale'] if 'scale' in normal_texture else 1
                    fallback_normal_color = (0,0,scale_factor)
                    scale_factor = (scale_factor, scale_factor, scale_factor, 1)
                    normal_components = {
                        'rgb':
                        {'sdf_type' : Sdf.ValueTypeNames.Normal3f, 'name': 'normal'}
                    }

                    self._convert_texture_to_usd(
                        pbr_mat=pbr_mat,
                        gltf_texture=material['normalTexture'],
                        gltf_texture_name='normalTexture',
                        color_components=normal_components,
                        scale_factor=scale_factor,
                        fallback_factor=fallback_normal_color,
                        material_path=material_path,
                        fallback_type=Sdf.ValueTypeNames.Normal3f,
                        primvar_st0_output=primvar_st0_output,
                        primvar_st1_output=primvar_st1_output
                    )

                if 'emissiveTexture' in material:
                    emissive_factor = material['emissiveFactor'] if 'emissiveFactor' in material else [0,0,0]
                    fallback_emissive_color = tuple(emissive_factor[0:3])
                    scale_emissive_factor = (emissive_factor[0], emissive_factor[1], emissive_factor[2], 1)
                    emissive_components = {
                        'rgb':
                        {'sdf_type' : Sdf.ValueTypeNames.Color3f, 'name': 'emissiveColor'}
                    }

                    self._convert_texture_to_usd(
                        pbr_mat=pbr_mat,
                        gltf_texture=material['emissiveTexture'],
                        gltf_texture_name='emissiveTexture',
                        color_components=emissive_components,
                        scale_factor=scale_emissive_factor,
                        fallback_factor=fallback_emissive_color,
                        material_path=material_path,
                        fallback_type=Sdf.ValueTypeNames.Color3f,
                        primvar_st0_output=primvar_st0_output,
                        primvar_st1_output=primvar_st1_output
                    )

                if pbr_specular_glossiness and 'diffuseTexture' in pbr_specular_glossiness:
                    base_color_factor = pbr_specular_glossiness['diffuseFactor'] if 'diffuseFactor' in pbr_specular_glossiness else (1,1,1,1)
                    fallback_base_color = (base_color_factor[0], base_color_factor[1], base_color_factor[2])
                    scale_base_color_factor = base_color_factor
                    base_color_components = {
                        'rgb':
                        {'sdf_type' : Sdf.ValueTypeNames.Color3f, 'name': 'diffuseColor'}
                    }

                    self._convert_texture_to_usd(
                        pbr_mat=pbr_mat,
                        gltf_texture=pbr_specular_glossiness['diffuseTexture'],
                        gltf_texture_name='diffuseTexture',
                        color_components=base_color_components,
                        scale_factor=scale_base_color_factor,
                        fallback_factor=fallback_base_color,
                        material_path=material_path,
                        fallback_type=Sdf.ValueTypeNames.Color3f,
                        primvar_st0_output=primvar_st0_output,
                        primvar_st1_output=primvar_st1_output
                    )


                if pbr_metallic_roughness and 'baseColorTexture' in pbr_metallic_roughness:
                    base_color_factor = pbr_metallic_roughness['baseColorFactor'] if 'baseColorFactor' in pbr_metallic_roughness else (1,1,1,1)
                    fallback_base_color = (base_color_factor[0], base_color_factor[1], base_color_factor[2])
                    scale_base_color_factor = base_color_factor
                    base_color_components = {
                        'rgb':
                        {'sdf_type' : Sdf.ValueTypeNames.Color3f, 'name': 'diffuseColor'}
                    }

                    self._convert_texture_to_usd(
                        pbr_mat=pbr_mat,
                        gltf_texture=pbr_metallic_roughness['baseColorTexture'],
                        gltf_texture_name='baseColorTexture',
                        color_components=base_color_components,
                        scale_factor=scale_base_color_factor,
                        fallback_factor=fallback_base_color,
                        material_path=material_path,
                        fallback_type=Sdf.ValueTypeNames.Color3f,
                        primvar_st0_output=primvar_st0_output,
                        primvar_st1_output=primvar_st1_output
                    )

                if pbr_metallic_roughness and 'metallicRoughnessTexture' in pbr_metallic_roughness:
                    metallic_roughness_texture_file = os.path.join(self.gltf_loader.root_dir, self.gltf_loader.json_data['images'][pbr_metallic_roughness['metallicRoughnessTexture']['index']]['uri'])
                    result = self.create_metallic_roughness_to_grayscale_images(metallic_roughness_texture_file)
                    metallic_factor = pbr_metallic_roughness['metallicFactor'] if 'metallicFactor' in pbr_metallic_roughness else 1.0
                    fallback_metallic = metallic_factor
                    scale_metallic = [metallic_factor] * 4
                    metallic_color_components = {
                        'b':
                        {'sdf_type' : Sdf.ValueTypeNames.Float, 'name': 'metallic'}
                    }

                    roughness_factor = pbr_metallic_roughness['roughnessFactor'] if 'roughnessFactor' in pbr_metallic_roughness else 1.0
                    fallback_roughness = roughness_factor
                    scale_roughness = [roughness_factor] * 4
                    roughness_color_components = {
                        'g':
                        {'sdf_type': Sdf.ValueTypeNames.Float, 'name': 'roughness'},
                    }


                    self._convert_texture_to_usd(
                        pbr_mat=pbr_mat,
                        gltf_texture=pbr_metallic_roughness['metallicRoughnessTexture'],
                        gltf_texture_name='metallicTexture',
                        color_components=metallic_color_components,
                        scale_factor=scale_metallic,
                        fallback_factor=fallback_metallic,
                        material_path=material_path,
                        fallback_type=Sdf.ValueTypeNames.Float,
                        primvar_st0_output=primvar_st0_output,
                        primvar_st1_output=primvar_st1_output
                    )

                    self._convert_texture_to_usd(
                        pbr_mat=pbr_mat,
                        gltf_texture=pbr_metallic_roughness['metallicRoughnessTexture'],
                        gltf_texture_name='roughnessTexture',
                        color_components=roughness_color_components,
                        scale_factor=scale_roughness,
                        fallback_factor=fallback_roughness,
                        material_path=material_path,
                        fallback_type=Sdf.ValueTypeNames.Float,
                        primvar_st0_output=primvar_st0_output,
                        primvar_st1_output=primvar_st1_output
                    )

                if pbr_specular_glossiness and 'specularGlossinessTexture' in pbr_specular_glossiness:
                    specular_glossiness_texture_file = os.path.join(self.gltf_loader.root_dir, self.gltf_loader.json_data['images'][pbr_specular_glossiness['specularGlossinessTexture']['index']]['uri'])
                    result = self.create_specular_glossiness_to_grayscale_images(specular_glossiness_texture_file)
                    specular_factor = tuple(pbr_specular_glossiness['specularFactor']) if 'specularFactor' in pbr_specular_glossiness else (1,1,1)
                    fallback_specular = specular_factor
                    scale_specular = [specular_factor[0], specular_factor[1], specular_factor[2], 1]
                    specular_color_components = {
                        'rgb':
                        {'sdf_type' : Sdf.ValueTypeNames.Color3f, 'name': 'specularColor'}
                    }

                    roughness_factor = 1 - pbr_specular_glossiness['glossiness'] if 'glossiness' in pbr_specular_glossiness else 0.0
                    fallback_roughness = roughness_factor
                    scale_roughness = [-1] * 4
                    glossiness_color_components = {
                        'a':
                        {'sdf_type': Sdf.ValueTypeNames.Float, 'name': 'roughness'},
                    }


                    self._convert_texture_to_usd(
                        pbr_mat=pbr_mat,
                        gltf_texture=pbr_specular_glossiness['specularGlossinessTexture'],
                        gltf_texture_name='specularTexture',
                        color_components=specular_color_components,
                        scale_factor=scale_specular,
                        fallback_factor=fallback_specular,
                        material_path=material_path,
                        fallback_type=Sdf.ValueTypeNames.Color3f,
                        primvar_st0_output=primvar_st0_output,
                        primvar_st1_output=primvar_st1_output
                    )

                    self._convert_texture_to_usd(
                        pbr_mat=pbr_mat,
                        gltf_texture=pbr_specular_glossiness['specularGlossinessTexture'],
                        gltf_texture_name='glossinessTexture',
                        color_components=glossiness_color_components,
                        scale_factor=[-1]*4,
                        fallback_factor=fallback_roughness,
                        material_path=material_path,
                        fallback_type=Sdf.ValueTypeNames.Float,
                        primvar_st0_output=primvar_st0_output,
                        primvar_st1_output=primvar_st1_output,
                        bias = (1.0, 1.0, 1.0, 1.0)
                    )

    def _convert_materials_to_preview_surface_new(self):
        """
        Converts the glTF materials to preview surfaces
        """
        self.usd_materials = []
        material_path_root = '/Materials'
        scope = UsdGeom.Scope.Define(self.stage, material_path_root)

        for i, material in enumerate(self.gltf_loader.get_materials()):
            usd_material = USDMaterial(self.stage, scope, i, self.gltf_loader)
            usd_material.convert_material_to_usd_preview_surface(material, self.output_dir)
            self.usd_materials.append(usd_material)


    def _convert_animation_to_usd(self, gltf_node, usd_node):
        animations = self.gltf_loader.get_animations()

        if len(animations) > 0:
            # only use first animation for now
            animation = animations[0]
            animation_channels = animation.get_animation_channels_for_node(gltf_node)

            if len(animation_channels) > 0:
                total_max_time = -999
                total_min_time = 999
                for channel in animation_channels:
                    min_max_time = self._create_usd_animation(usd_node, channel)
                    total_max_time = max(total_max_time, min_max_time.max)
                    total_min_time = min(total_min_time, min_max_time.min)

                self.stage.SetStartTimeCode(total_min_time)
                self.stage.SetEndTimeCode(total_max_time)
                self.stage.SetTimeCodesPerSecond(self.fps)


    def _convert_skin_to_usd(self, gltf_node, gltf_primitive, usd_node, usd_mesh):
        """Converts a glTF skin to a UsdSkel

        Arguments:
            gltf_node {dict} -- glTF node
            node_index {int} -- index of the glTF node
            usd_parent_node {UsdPrim} -- parent node of the usd node
            usd_mesh {[type]} -- [description]
        """
        skel_binding_api = UsdSkel.BindingAPI(usd_mesh)
        gltf_skin = gltf_node.get_skin()
        gltf_joint_names = [GLTF2USDUtils.convert_to_usd_friendly_node_name(joint.get_name()) for joint in gltf_skin.get_joints()]
        usd_joint_names = [Sdf.Path(self._get_usd_joint_hierarchy_name(joint, gltf_skin.get_root_joint())) for joint in gltf_skin.get_joints()]
        skeleton = self._create_usd_skeleton(gltf_skin, usd_node, usd_joint_names)
        skeleton_animation = self._create_usd_skeleton_animation(gltf_skin, skeleton, usd_joint_names)

        parent_path = usd_node.GetPath()

        bind_matrices = []
        rest_matrices = []

        skeleton_root = self.stage.GetPrimAtPath(parent_path)
        skel_binding_api = UsdSkel.BindingAPI(usd_mesh)
        skel_binding_api.CreateGeomBindTransformAttr(Gf.Matrix4d(((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))))
        skel_binding_api.CreateSkeletonRel().AddTarget(skeleton.GetPath())
        
        skel_binding_api.CreateAnimationSourceRel().AddTarget(skeleton_animation.GetPath())
        bind_matrices = self._compute_bind_transforms(gltf_skin)
        gltf_root_node_name = GLTF2USDUtils.convert_to_usd_friendly_node_name(gltf_skin.get_root_joint().get_name())

        primitive_attributes = gltf_primitive.get_attributes()

        if 'WEIGHTS_0' in primitive_attributes and 'JOINTS_0' in primitive_attributes:
            total_vertex_weights = primitive_attributes['WEIGHTS_0'].get_data()
            total_vertex_joints = primitive_attributes['JOINTS_0'].get_data()
            total_joint_indices = []
            total_joint_weights = []

            for joint_indices, weights in zip(total_vertex_joints, total_vertex_weights):
                for joint_index, weight in zip(joint_indices, weights):
                    total_joint_indices.append(joint_index)
                    total_joint_weights.append(weight)

            joint_indices_attr = skel_binding_api.CreateJointIndicesPrimvar(False, 4).Set(total_joint_indices)
            total_joint_weights = Vt.FloatArray(total_joint_weights)
            UsdSkel.NormalizeWeights(total_joint_weights, 4)
            joint_weights_attr = skel_binding_api.CreateJointWeightsPrimvar(False, 4).Set(total_joint_weights)

    
    def _compute_bind_transforms(self, gltf_skin):
        """Compute the bind matrices from the skin

        Arguments:
            gltf_skin {Skin} -- glTF skin

        Returns:
            [list] -- List of bind matrices
        """

        bind_matrices = []
        inverse_bind_matrices = gltf_skin.get_inverse_bind_matrices()

        for matrix in inverse_bind_matrices:
            bind_matrix = self._convert_to_usd_matrix(matrix)
            bind_matrices.append(bind_matrix.GetInverse())

        return bind_matrices

    def _convert_to_usd_matrix(self, matrix):
        """Converts a glTF matrix to a Usd Matrix

        Arguments:
            matrix {[type]} -- [description]

        Returns:
            [type] -- [description]
        """

        return Gf.Matrix4d(
            matrix[0], matrix[1], matrix[2], matrix[3],
            matrix[4], matrix[5], matrix[6], matrix[7],
            matrix[8], matrix[9], matrix[10], matrix[11],
            matrix[12], matrix[13], matrix[14], matrix[15]
        )

    def _compute_rest_matrix(self, gltf_node):
        """
        Compute the rest matrix from a glTF node.
        The translation, rotation and scale are combined into a transformation matrix

        Returns:
            Matrix4d -- USD matrix
        """

        xform_matrix = None
        matrix = gltf_node.get_matrix()
        if matrix != None:
            xform_matrix = self._convert_to_usd_matrix(matrix)
            return xform_matrix
        else:
            usd_scale = Gf.Vec3h(1,1,1)
            usd_rotation = Gf.Quatf()
            usd_translation = Gf.Vec3f(0,0,0)

            scale = gltf_node.get_scale()
            usd_scale = Gf.Vec3h(scale[0], scale[1], scale[2])

            rotation = gltf_node.get_rotation()
            usd_rotation = Gf.Quatf(rotation[3], rotation[0], rotation[1], rotation[2])

            translation = gltf_node.get_translation()
            usd_translation = Gf.Vec3f(translation[0], translation[1], translation[2])

        return UsdSkel.MakeTransform(usd_translation, usd_rotation, usd_scale)


    def _create_usd_animation(self, usd_node, animation_channel):
        """Converts a glTF animation to a USD animation

        Arguments:
            usd_node {[type]} -- usd node
            animation_channel {AnimationMap} -- map of animation target path and animation sampler indices

        Returns:
            [type] -- [description]
        """

        sampler = animation_channel.get_sampler()
        
    
        max_time = int(round(sampler.get_input_max()[0] * self.fps))
        min_time = int(round(sampler.get_input_min()[0] * self.fps))
        input_keyframes = sampler.get_input_data()
        output_keyframes = sampler.get_output_data()

        num_values = sampler.get_output_count() / sampler.get_input_count()
        (transform, convert_func) = self._get_keyframe_conversion_func(usd_node, animation_channel)

        for i, keyframe in enumerate(input_keyframes):
            convert_func(transform, int(round(keyframe * self.fps)), output_keyframes, i, num_values)

        MinMaxTime = collections.namedtuple('MinMaxTime', ('max', 'min'))
        return MinMaxTime(max=max_time, min=min_time)

    def _get_keyframe_conversion_func(self, usd_node, animation_channel):
        """Convert glTF key frames to USD animation key frames

        Arguments:
            usd_node {UsdPrim} -- USD node to apply animations to
            animation_channel {obj} -- glTF animation

        Raises:
            Exception -- [description]

        Returns:
            [func] -- USD animation conversion function
        """

        path = animation_channel.get_target().get_path()

        def convert_translation(transform, time, output, i, _):
            value = output[i]
            transform.Set(time=time, value=(value[0], value[1], value[2]))

        def convert_scale(transform, time, output, i, _):
            value = output[i]
            transform.Set(time=time, value=(value[0], value[1], value[2]))

        def convert_rotation(transform, time, output, i, _):
            value = output[i]
            transform.Set(time=time, value=Gf.Quatf(value[3], value[0], value[1], value[2]))

        def convert_weights(transform, time, output, i, values_per_step):
            start = i * values_per_step
            end = start + values_per_step
            values = output[start:end]
            value = list(map(lambda x: round(x, 5) + 0, values))
            transform.Set(time=time, value=value)

        if path == 'translation':
            return (usd_node.AddTranslateOp(opSuffix='translate'), convert_translation)
        elif path == 'rotation':
            return (usd_node.AddOrientOp(opSuffix='rotate'), convert_rotation)
        elif path == 'scale':
            return (usd_node.AddScaleOp(opSuffix='scale'), convert_scale)
        elif path == 'weights':
            prim = usd_node.GetPrim().GetChild("skeleton_root").GetChild("skel").GetChild("anim")
            anim_attr = prim.GetAttribute('blendShapeWeights')
            return (anim_attr, convert_weights)
        else:
            raise Exception('Unsupported animation target path! {}'.format(path))


    def unpack_textures_to_grayscale_images(self, image_path, color_components):
        image_base_name = ntpath.basename(image_path)
        texture_name = image_base_name
        for color_component, sdf_type in color_components.iteritems():
            if color_component == 'rgb':
                pass
            else:
                img = Image.open(image_path)
                if img.mode == 'P':
                    img = img.convert('RGB')
                if img.mode == 'RGB':
                    occlusion, roughness, metallic = img.split()
                    if color_component == 'r':
                        texture_name = 'Occlusion_{}'.format(image_base_name)
                        occlusion.save(os.path.join(self.output_dir, texture_name))
                    elif color_component == 'g':
                        texture_name = 'Roughness_{}'.format(image_base_name)
                        roughness.save(os.path.join(self.output_dir, texture_name))
                    elif color_component == 'b':
                        texture_name = 'Metallic_{}'.format(image_base_name)
                        metallic.save(os.path.join(self.output_dir, texture_name))
                elif img.mode == 'RGBA':
                    if color_component == 'a':
                        texture_name = 'Glossiness_{}'.format(image_base_name)
                        img.getchannel('A').save(os.path.join(self.output_dir, texture_name))
                    else:
                        raise Exception('unrecognized image type!: {}'.format(img.mode))
                elif img.mode == 'L':
                    #already single channel
                    pass
                else:
                    raise Exception('Unsupported image type!: {}'.format(img.mode))


        return texture_name

    def convert_metallic_roughness_texture_to_usd(self, metallic_roughness_texture):
        image_path = metallic_roughness_texture.get_image_path()
        image_base_name = ntpath.basename(image_path)
        texture_name = image_base_name
        img = Image.open(image_path)
        if img.mode == 'P':
            img = img.convert('RGB')

        _, metallic, roughness = img.split()
        roughness_texture_name = 'Roughness_{}'.format(image_base_name)
        roughness.save(os.path.join(self.output_dir, texture_name))

        metallic_texture_name = 'Metallic_{}'.format(image_base_name)
        metallic.save(os.path.join(self.output_dir, texture_name))

        return {'roughness': roughness_texture_name, 'metallic': metallic_texture_name}

    def convert_specular_glossiness_texture_to_usd(self, specular_glossiness_texture):
        image_path = specular_glossiness_texture.get_image_path()
        image_base_name = ntpath.basename(image_path)
        texture_name = image_base_name
        img = Image.open(image_path)
        specular_texture_name = os.path.join(self.output_dir, 'Specular_{}'.format(image_base_name))
        glossiness_texture_name = os.path.join(self.output_dir, 'Glossiness_{}'.format(image_base_name))
        if img.mode == 'P':
            #convert paletted image to RGBA
            img = img.convert('RGBA')
        
        #get specular
        img.convert('RGB').save(specular_texture_name)
        #get glossiness
        img.getchannel('A').save(glossiness_texture_name)
        #channels[3].save(glossiness_texture_name)
        

        return {'specular': specular_texture_name, 'glossiness': glossiness_texture_name}



    def create_metallic_roughness_to_grayscale_images(self, image):
        image_base_name = ntpath.basename(image)
        roughness_texture_name = os.path.join(self.output_dir, 'Roughness_{}'.format(image_base_name))
        metallic_texture_name = os.path.join(self.output_dir, 'Metallic_{}'.format(image_base_name))

        img = Image.open(image)

        if img.mode == 'P':
            #convert paletted image to RGB
            img = img.convert('RGB')
        if img.mode == 'RGB':
            channels = img.split()
            #get roughness
            channels[1].save(roughness_texture_name)
            #get metalness
            channels[2].save(metallic_texture_name)

            return {'metallic': metallic_texture_name, 'roughness': roughness_texture_name}

    def create_specular_glossiness_to_grayscale_images(self, image):
        image_base_name = ntpath.basename(image)
        specular_texture_name = os.path.join(self.output_dir, 'Specular_{}'.format(image_base_name))
        glossiness_texture_name = os.path.join(self.output_dir, 'Glossiness_{}'.format(image_base_name))

        img = Image.open(image)

        if img.mode == 'P':
            #convert paletted image to RGBA
            img = img.convert('RGBA')
        if img.mode == 'RGBA':
            #get specular
            img.convert('RGB').save(specular_texture_name)
            #get glossiness
            img.getchannel('A').save(glossiness_texture_name)
            #channels[3].save(glossiness_texture_name)

            return {'specular': specular_texture_name, 'glossiness': glossiness_texture_name}

    '''
    Converts a glTF texture to USD
    '''
    def _convert_texture_to_usd(self, primvar_st0_output, primvar_st1_output, pbr_mat, gltf_texture, gltf_texture_name, color_components, scale_factor, fallback_factor, material_path, fallback_type, bias=None):
        image_name = gltf_texture if (isinstance(gltf_texture, basestring)) else self.images[gltf_texture['index']]
        image_path = os.path.join(self.output_dir, image_name)
        texture_index = int(gltf_texture['index'])
        texture = self.gltf_loader.json_data['textures'][texture_index]
        wrap_modes = self._get_texture__wrap_modes(texture)
        texture_shader = UsdShade.Shader.Define(self.stage, material_path.AppendChild(gltf_texture_name))
        texture_shader.CreateIdAttr("UsdUVTexture")

        texture_shader.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set(wrap_modes['wrapS'])
        texture_shader.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set(wrap_modes['wrapT'])
        if bias:
            texture_shader.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(bias)

        texture_name = self.unpack_textures_to_grayscale_images(image_path, color_components)
        file_asset = texture_shader.CreateInput('file', Sdf.ValueTypeNames.Asset)
        file_asset.Set(texture_name)

        for color_params, usd_color_params in color_components.iteritems():
            sdf_type = usd_color_params['sdf_type']
            texture_shader_output = texture_shader.CreateOutput(color_params, sdf_type)
            pbr_mat_texture = pbr_mat.CreateInput(usd_color_params['name'], sdf_type)
            pbr_mat_texture.ConnectToSource(texture_shader_output)

        texture_shader_input = texture_shader.CreateInput('st', Sdf.ValueTypeNames.Float2)
        texture_shader_fallback = texture_shader.CreateInput('fallback', fallback_type)
        texture_shader_fallback.Set(fallback_factor)
        if 'texCoord' in gltf_texture and gltf_texture['texCoord'] == 1:
            texture_shader_input.ConnectToSource(primvar_st1_output)
        else:
            texture_shader_input.ConnectToSource(primvar_st0_output)

        scale_vector = texture_shader.CreateInput('scale', Sdf.ValueTypeNames.Float4)
        scale_vector.Set((scale_factor[0], scale_factor[1], scale_factor[2], scale_factor[3]))

    def convert(self):
        if hasattr(self, 'gltf_loader'):
            self._convert_images_to_usd()
            self._convert_materials_to_preview_surface_new()
            self.convert_nodes_to_xform()

def check_usd_compliance(rootLayer, arkit=False):
    checker = UsdUtils.ComplianceChecker(rootLayer, arkit=arkit, skipARKitRootLayerCheck=False)
    errors = checker.GetErrors()
    failedChecks = checker.GetFailedChecks()
    for msg in errors + failedChecks:
        print(msg)
    return len(errors) == 0 and len(failedChecks) == 0


def convert_to_usd(gltf_file, usd_file, fps, scale, arkit=False, verbose=False):
    """Converts a glTF file to USD

    Arguments:
        gltf_file {str} -- path to glTF file
        usd_file {str} -- path to write USD file

    Keyword Arguments:
        verbose {bool} -- [description] (default: {False})
    """

    usd = GLTF2USD(gltf_file=gltf_file, usd_file=usd_file, fps=fps, scale=scale, verbose=verbose)
    if usd.stage:
        asset = usd.stage.GetRootLayer()
        usd.logger.info('Conversion complete!')

        asset.Save()
        usd.logger.info('created {}'.format(asset.realPath))

        if usd_file.endswith('.usdz') or usd_file.endswith('.usdc'):
            usdc_file = '%s.%s' % (os.path.splitext(usd_file)[0], 'usdc')
            asset.Export(usdc_file, args=dict(format='usdc'))
            usd.logger.info('created {}'.format(usdc_file))

        if usd_file.endswith('.usdz'):
            r = Ar.GetResolver()
            resolvedAsset = r.Resolve(usdc_file)
            context = r.CreateDefaultContextForAsset(resolvedAsset)

            success = check_usd_compliance(resolvedAsset, arkit=args.arkit)
            with Ar.ResolverContextBinder(context):
                if arkit and not success:
                    usd.logger.warning('USD is not ARKit compliant')
                    return

                success = UsdUtils.CreateNewUsdzPackage(usdc_file, usd_file) and success

                if success:
                    usd.logger.info('created package {} with contents:'.format(usd_file))
                    zip_file = Usd.ZipFile.Open(usd_file)
                    file_names = zip_file.GetFileNames()
                    for file_name in file_names:
                        usd.logger.info('\t{}'.format(file_name))
                else:
                    usd.logger.error('could not create {}'.format(usd_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert glTF to USD')
    parser.add_argument('--gltf', '-g', action='store', dest='gltf_file', help='glTF file (in .gltf format)', required=True)
    parser.add_argument('--fps', action='store', dest='fps', help='The frames per second for the animations', type=float, default=24.0)
    parser.add_argument('--output', '-o', action='store', dest='usd_file', help='destination to store generated .usda file', required=True)
    parser.add_argument('--verbose', '-v', action='store_true', dest='verbose', help='Enable verbose mode')
    parser.add_argument('--scale', '-s', action='store', dest='scale', help='Scale the resulting USDA', type=float, default=100)
    parser.add_argument('--arkit', action='store_true', dest='arkit', help='Check USD with ARKit compatibility before making USDZ file')
    args = parser.parse_args()

    if args.gltf_file:
        convert_to_usd(args.gltf_file, args.usd_file, args.fps, args.scale, args.arkit, args.verbose)
