from jsonschema import validate, Draft4Validator

CONFIG_SCHEMA={
"type": "object",
    "properties": {
        "xi": {"type": "number"},
        "xf": {"type": "number"},
        "fi": {"type": "number"},
        "ff": {"type": "number"},
        "nfre": {"type": "number"},
        "ptx": {"type": "string"},
        "ifbw": {"type": "number"}
    }
}

config={
    "xi": 0.0,
    "yf": 1.4,
    "fi": 15.5,
    "ff": 15.6,
    "nfre": '2',
    "ptx": 1,
    "ifbw": 'a',
    "beam_angle": 150
}

v=Draft4Validator(CONFIG_SCHEMA)
errors=sorted(v.iter_errors(config), key=lambda e: e.path)

for error in errors:
    print error.message, ''.join(error.path)

for error in errors:
    for suberror in sorted(error.context, key=lambda e: e.schema_path):
        print list(suberror.schema_path)
