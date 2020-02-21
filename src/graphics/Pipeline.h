#pragma once

#include <cstdint>

#include <memory>
#include <string_view>
#include <glm/fwd.hpp>

class Buffer;

/// Represents a GPU pipeline with all attribute which would cause recompilation
/// inside the driver. Using a pipeline object for each collection of state
/// reduces recompilation in the drivers due to these state changes.
class Pipeline {
  public:
    struct GraphicsCreateInfo {
        uint32_t ViewportWidth;
        uint32_t ViewportHeight;
        std::string_view VertexShaderSource;
        std::string_view FragmentShaderSource;
        float LineWidth;
        std::string_view DebugName;
    };
    struct ComputeCreateInfo {
        std::string_view ShaderSource;
        std::string_view DebugName;
    };

    /// Bind pipeline with which to draw
    virtual void bind() = 0;
    /// Upload a uniform: data which is shared with all shader cores during dispatch.
    virtual bool setUniform(const std::string_view& uniform_name, const glm::vec3& uniform) = 0;
    virtual bool setUniform(const std::string_view& uniform_name, const glm::vec4& uniform) = 0;
    virtual bool setUniform(const std::string_view& uniform_name, const glm::mat4& uniform) = 0;
    virtual bool setUniform(const std::string_view& uniform_name, const float& uniform) = 0;
    virtual bool setUniform(const std::string_view& uniform_name, uint32_t index, const Buffer& uniform) = 0;

    /// A factory function in the impl class allows for an error to return null
    static std::unique_ptr<Pipeline> create(const GraphicsCreateInfo& info);
    static std::unique_ptr<Pipeline> create(const ComputeCreateInfo& info);
};
