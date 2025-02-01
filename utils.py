def flatten(nested_list):
            flattened = []
            for sublist in nested_list:
                flattened.extend(sublist)
                
            return flattened