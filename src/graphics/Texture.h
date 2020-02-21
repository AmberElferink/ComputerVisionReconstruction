#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

namespace cv
{
class Mat;
}

/// Wrapper around OpenGL textures, OpenCV-OpenGL interop and samplers
/// Allows upload of data but not resizing or modifying texture attributes
/// once they are created. These are expensive operations which are not
/// desired.
class Texture
{
public:
    enum class Format
    {
        r_snorm,
        rg_snorm,
        rgb_snorm,
        rgba_snorm,

        r8f,
        rg8f,
        rgb8f,
        rgba8f,

        r16f,
        rg16f,
        rgb16f,
        rgba16f,

        r32f,
        rg32f,
        rgb32f,
        rgba32f,

        r8i,
        rg8i,
        rgb8i,
        rgba8i,

        r16i,
        rg16i,
        rgb16i,
        rgba16i,

        r32i,
        rg32i,
        rgb32i,
        rgba32i,

        r8u,
        rg8u,
        rgb8u,
        rgba8u,

        r16u,
        rg16u,
        rgba16u,
        rgb16u,

        r32u,
        rg32u,
        rgb32u,
        rgba32u,
    };
    struct CreateInfo {
        uint32_t Width;
        uint32_t Height;
        uint32_t Depth;
        Format DataFormat;
        std::string_view DebugName;
    };
    /// Factory function for creating a texture with a specific dimension
    /// only one color format is used which is why it is not an argument.
    static std::unique_ptr<Texture> create(const CreateInfo& info);
    virtual ~Texture();

    /// Bind a texture for sampling in a pipeline
    void bind();

    void upload(const void* data, uint32_t size);

private:
    /// Private unique constructor forcing the use of factory function which
    /// can return null unlike constructor.
    Texture(uint32_t handle, uint32_t target, const CreateInfo& info);

    const uint32_t handle_;
    const uint32_t target_;
    const CreateInfo info_;
};
