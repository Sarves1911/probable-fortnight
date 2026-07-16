#include "loader.hpp"

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>

static std::string read_text_file(const std::string &path)
{
    std::ifstream file(path);

    if (!file)
    {
        throw std::runtime_error("Could not open metadata file: " + path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

static std::vector<int> parse_shape(const std::string &json)
{
    std::regex shape_regex("\"shape\"\\s*:\\s*\\[([^\\]]+)\\]");
    std::smatch match;

    if (!std::regex_search(json, match, shape_regex))
    {
        throw std::runtime_error("Could not parse shape from metadata");
    }

    std::string inside = match[1].str();

    std::regex int_regex("-?\\d+");
    std::sregex_iterator begin(inside.begin(), inside.end(), int_regex);
    std::sregex_iterator end;

    std::vector<int> shape;

    for (auto it = begin; it != end; ++it)
    {
        shape.push_back(std::stoi(it->str()));
    }

    return shape;
}

static float parse_float_field(const std::string &json, const std::string &field)
{
    std::regex field_regex("\"" + field + "\"\\s*:\\s*([-+0-9.eE]+)");
    std::smatch match;

    if (!std::regex_search(json, match, field_regex))
    {
        throw std::runtime_error("Could not parse float field: " + field);
    }

    return std::stof(match[1].str());
}

static int parse_int_field_or_default(
    const std::string &json,
    const std::string &field,
    int default_value)
{
    std::regex field_regex("\"" + field + "\"\\s*:\\s*(-?\\d+)");
    std::smatch match;

    if (!std::regex_search(json, match, field_regex))
    {
        return default_value;
    }

    return std::stoi(match[1].str());
}

Tensor load_tensor_i8(const std::string &bin_path, const std::string &meta_path)
{
    std::string json = read_text_file(meta_path);

    Tensor tensor;
    tensor.shape = parse_shape(json);
    tensor.scale = parse_float_field(json, "scale");
    tensor.zero_point = parse_int_field_or_default(json, "zero_point", 0);

    int expected_count = tensor.numel();
    tensor.data.resize(expected_count);

    std::ifstream file(bin_path, std::ios::binary);

    if (!file)
    {
        throw std::runtime_error("Could not open binary tensor file: " + bin_path);
    }

    file.read(reinterpret_cast<char *>(tensor.data.data()), expected_count * sizeof(int8_t));

    std::streamsize bytes_read = file.gcount();
    std::streamsize expected_bytes = expected_count * sizeof(int8_t);

    if (bytes_read != expected_bytes)
    {
        throw std::runtime_error(
            "Short read from " + bin_path +
            ". Expected " + std::to_string(expected_bytes) +
            " bytes, got " + std::to_string(bytes_read));
    }

    return tensor;
}