class Node:
    """Create a glTF node object
    """

    def __init__(self, node_dict, node_index):
        self._name = None
        self._parent = None
        self._name = node_dict['name'] if ('name' in node_dict) else 'node_{}'.format(node_index)
        self._node_index = node_index
        self._matrix = node_dict['matrix'] if ('matrix' in node_dict) else None
        self._translation = node_dict['translation'] if ('translation' in node_dict) else [0,0,0]
        self._rotation = node_dict['rotation'] if ('rotation' in node_dict) else [0,0,0,1]     
        self._scale = node_dict['scale'] if ('scale' in node_dict) else [1,1,1]   
        
        self._mesh_index = node_dict['mesh'] if ('mesh' in node_dict) else None
        self._mesh = None
        
        self._children_indices = node_dict['children'] if ('children' in node_dict) else []
        self._children = []

    def get_name(self):
        return self._name

    def get_translation(self):
        return self._translation

    def get_rotation(self):
        return self._rotation

    def get_scale(self):
        return self._scale

    def get_children(self):
        return self._children

    def get_parent(self):
        return self._parent

    def get_matrix(self):
        return self._matrix

    def get_index(self):
        return self._node_index