def data_type_inspector(data_list):
    print("Analyzing Data Types...\n")
    for item in data_list:
        print(f"Value: 
              {item!r} | Type: {type(item).__name__}")

if __name__ == "__main__":
    sample_data = [42, 3.14, "hello", True, None,
                    [1, 2], {"key": "value"}, \
                        (1, 2), {1, 2}]
    data_type_inspector(sample_data)
