from morph_tool import convert


def run_conversion():
    convert("morphologies/n173.CNG.swc", "morphologies/n173.swc", sanitize=True, single_point_soma=True)


if __name__ == "__main__":
    run_conversion()
