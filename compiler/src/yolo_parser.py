import onnx
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser")

    parser.add_argument('--onnx_path', type=str, default=None)
    
    arg = parser.parse_args()

    onnx_file_path = arg.onnx_path

    model = onnx.load(onnx_file_path)

    input_nodes = model.graph.input

    for input_node in input_nodes:
        input_name = input_node.name
        input_shape = input_node.type.tensor_type.shape.dim
    
        dim_info = list()
        for dim in input_shape:
            dim_size = dim.dim_value
            dim_name = dim.dim_param
            dim_info.append((dim_size, dim_name))

            for i, (size, name) in enumerate(dim_info):
                if i == 2:
                    height = size
                if i == 3:
                    width = size


    #dump
    f = open("src/input_size.txt", "w")
    f.write(f"{width};\n")
    f.write(f"{height};\n")
    f.close()
