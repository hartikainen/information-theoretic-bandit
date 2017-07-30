import json


class ObjectEncoder(json.JSONEncoder):
  def default(self, obj):
    if hasattr(obj, "to_json"):
      return self.default(obj.to_json())

    return obj

def dump_result(filename, results):
  data = []
  try:
    with open(filename, "r") as f:
      data = json.load(f)
  except FileNotFoundError:
    pass
  finally:
    data.append(results)
    with open(filename, "w") as f:
      json.dump(data, f, sort_keys=True,
                indent=4, separators=(',', ': '),
                cls=ObjectEncoder)

def load_result(filename):
  with open(filename, "r") as f:
    results = json.load(f)

  # TODO: change this
  return results[0]
