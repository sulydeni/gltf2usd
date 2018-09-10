import base64
from io import BytesIO
import os

from PIL import Image

class GLTFImage:
    def __init__(self, image_entry, image_index, gltf_loader):
        if image_entry['uri'].startswith('data:image'):
            uri_data = image_entry['uri'].split(',')[1]
            img = Image.open(BytesIO(base64.b64decode(uri_data)))

            # NOTE: image might not have a name
            self._name = image_entry['name'] if 'name' in image_entry else 'image{}.{}'.format(image_index, img.format.lower())
            self._image_path = os.path.join(gltf_loader.root_dir, self._name)
            img.save(self._image_path)
        else:
            self._name = image_entry['name'] if ('name' in image_entry) else 'image_{}'.format(image_index)
            self._uri = image_entry['uri']
            self._image_path = os.path.join(gltf_loader.root_dir, self._name)