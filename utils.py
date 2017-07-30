import json

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
                      indent=4, separators=(',', ': '))
