class Skin:
    """Represents a glTF Skin
    """

    def __init__(self, gltf2_loader, skin_entry):
        self._inverse_bind_matrices = self._init_inverse_bind_matrices(gltf2_loader, skin_entry)
        self._joints = [gltf2_loader.nodes[joint_index] for joint_index in skin_entry['joints']]
        self._root_skeleton = self._init_root_skeleton(gltf2_loader, skin_entry)

    def _init_inverse_bind_matrices(self, gltf2_loader, skin_entry):
        inverse_bind_matrices = []
        if 'inverseBindMatrices' in skin_entry:
            inverse_bind_matrices = gltf2_loader.get_data(accessor=gltf2_loader.json_data['accessors'][skin_entry['inverseBindMatrices']])
        
        return inverse_bind_matrices

    def get_inverse_bind_matrices(self):
        return self._inverse_bind_matrices

    def get_joints(self):
        return self._joints

    def _init_root_skeleton(self, gltf2_loader, skin_entry):
        if 'skeleton' in skin_entry:
            return gltf2_loader.nodes[skin_entry['skeleton']]

        else:
            joint = gltf2_loader.nodes[skin_entry['joints'][0]]
            print(joint)

            parent = joint.get_parent()
            while (parent != None):
                joint = parent
                parent = joint.get_parent()

            return joint

    def get_bind_transforms(self):
        pass

    def get_rest_transforms(self):
        pass
    
    def get_joint_names(self):
        pass

    def get_root_joint(self):
        return self._root_skeleton