import re 
import json
def get_content_from_json(json_data):
    json_str = re.search(r"```json\n(.*?)\n```", json_data.content, re.DOTALL).group(1)
    data = json.loads(json_str)
    return data