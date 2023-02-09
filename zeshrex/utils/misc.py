from types import SimpleNamespace


def convert_dict_to_namespace(dictionary: dict) -> SimpleNamespace:
    """
    Converts dictionary into SimpleNamespace object to make a dot
    notation for accessing a variable. Recursively traverses the dictionary and
    converts all sub-dictionaries into a SimpleNamespace.

    :param dictionary: Dictionary to convert to a SimpleNamespace object.
    :return: SimpleNamespace object constructed from the source dictionary.
    """
    prepared_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            converted_value = convert_dict_to_namespace(value)
            prepared_dict[key] = converted_value
        else:
            prepared_dict[key] = value
    return SimpleNamespace(**prepared_dict)
