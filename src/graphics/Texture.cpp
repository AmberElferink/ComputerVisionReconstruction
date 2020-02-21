#include "Texture.h"

#include <cassert>
#include <array>
#include <glad/glad.h>

struct TextureFormatLookUp
{
  const uint32_t format;
  const uint32_t internal_format;
  const uint32_t type;
  const uint8_t size;
};

std::array<TextureFormatLookUp, 40> TextureFormatLookUpTable = {
    /* r_snorm    */ TextureFormatLookUp{ GL_RED, GL_R8_SNORM, GL_BYTE, 1 },
    /* rg_snorm   */ TextureFormatLookUp{ GL_RG, GL_RG8_SNORM, GL_BYTE, 2 },
    /* rgb_snorm  */ TextureFormatLookUp{ GL_RGB, GL_RGB8_SNORM, GL_BYTE, 3 },
    /* rgba_snorm */ TextureFormatLookUp{ GL_RGBA, GL_RGBA8_SNORM, GL_BYTE, 4 },

    /* r8f        */ TextureFormatLookUp{ GL_RED, GL_R8, GL_UNSIGNED_BYTE, 1 },
    /* rg8f       */ TextureFormatLookUp{ GL_RG, GL_RG8, GL_UNSIGNED_BYTE, 2 },
    /* rgb8f      */ TextureFormatLookUp{ GL_RGB, GL_RGB8, GL_UNSIGNED_BYTE, 3 },
    /* rgba8f     */ TextureFormatLookUp{ GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE, 4 },

    /* r16f       */ TextureFormatLookUp{ GL_RED, GL_R16F, GL_HALF_FLOAT, 2 },
    /* rg16f      */ TextureFormatLookUp{ GL_RG, GL_RG16F, GL_HALF_FLOAT, 4 },
    /* rgb16f     */ TextureFormatLookUp{ GL_RGB, GL_RGB16F, GL_HALF_FLOAT, 6 },
    /* rgba16f    */ TextureFormatLookUp{ GL_RGBA, GL_RGBA16F, GL_HALF_FLOAT, 8 },

    /* r32f       */ TextureFormatLookUp{ GL_RED, GL_R32F, GL_FLOAT, 4 },
    /* rg32f      */ TextureFormatLookUp{ GL_RG, GL_RG32F, GL_FLOAT, 8 },
    /* rgb32f     */ TextureFormatLookUp{ GL_RGB, GL_RGB32F, GL_FLOAT, 12 },
    /* rgba32f    */ TextureFormatLookUp{ GL_RGBA, GL_RGBA32F, GL_FLOAT, 16 },

    /* r8i        */ TextureFormatLookUp{ GL_RED_INTEGER, GL_R8I, GL_BYTE, 1 },
    /* rg8i       */ TextureFormatLookUp{ GL_RG_INTEGER, GL_RG8I, GL_BYTE, 2 },
    /* rgb8i      */ TextureFormatLookUp{ GL_RGB_INTEGER, GL_RGB8I, GL_BYTE, 3 },
    /* rgba8i     */ TextureFormatLookUp{ GL_RGBA_INTEGER, GL_RGBA8I, GL_BYTE, 4 },

    /* r16i       */ TextureFormatLookUp{ GL_RED_INTEGER, GL_R16I, GL_SHORT, 2 },
    /* rg16i      */ TextureFormatLookUp{ GL_RG_INTEGER, GL_RG16I, GL_SHORT, 4 },
    /* rgb16i     */ TextureFormatLookUp{ GL_RGB_INTEGER, GL_RGB16I, GL_SHORT, 6 },
    /* rgba16i    */ TextureFormatLookUp{ GL_RGBA_INTEGER, GL_RGBA16I, GL_SHORT, 8 },

    /* r32i       */ TextureFormatLookUp{ GL_RED_INTEGER, GL_R32I, GL_INT, 4 },
    /* rg32i      */ TextureFormatLookUp{ GL_RG_INTEGER, GL_RG32I, GL_INT, 8 },
    /* rgb32i     */ TextureFormatLookUp{ GL_RGB_INTEGER, GL_RGB32I, GL_INT, 12 },
    /* rgba32i    */ TextureFormatLookUp{ GL_RGBA_INTEGER, GL_RGBA32I, GL_INT, 16 },

    /* r8u        */ TextureFormatLookUp{ GL_RED_INTEGER, GL_R8UI, GL_UNSIGNED_BYTE, 1 },
    /* rg8u       */ TextureFormatLookUp{ GL_RG_INTEGER, GL_RG8UI, GL_UNSIGNED_BYTE, 2 },
    /* rgb8u      */ TextureFormatLookUp{ GL_RGB_INTEGER, GL_RGB8UI, GL_UNSIGNED_BYTE, 3 },
    /* rgba8u     */ TextureFormatLookUp{ GL_RGBA_INTEGER, GL_RGBA8UI, GL_UNSIGNED_BYTE, 4 },

    /* r16u       */ TextureFormatLookUp{ GL_RED_INTEGER, GL_R16UI, GL_UNSIGNED_SHORT, 2 },
    /* rg16u      */ TextureFormatLookUp{ GL_RG_INTEGER, GL_RG16UI, GL_UNSIGNED_SHORT, 4 },
    /* rgb16u     */ TextureFormatLookUp{ GL_RGB_INTEGER, GL_RGB16UI, GL_UNSIGNED_SHORT, 6 },
    /* rgba16u    */ TextureFormatLookUp{ GL_RGBA_INTEGER, GL_RGBA16UI, GL_UNSIGNED_SHORT, 8 },

    /* r32u       */ TextureFormatLookUp{ GL_RED_INTEGER, GL_R32UI, GL_UNSIGNED_INT, 4 },
    /* rg32u      */ TextureFormatLookUp{ GL_RG_INTEGER, GL_RG32UI, GL_UNSIGNED_INT, 8 },
    /* rgb32u     */ TextureFormatLookUp{ GL_RGB_INTEGER, GL_RGB32UI, GL_UNSIGNED_INT, 12 },
    /* rgba32u    */ TextureFormatLookUp{ GL_RGBA_INTEGER, GL_RGBA32UI, GL_UNSIGNED_INT, 16 },
};

std::unique_ptr<Texture> Texture::create(const CreateInfo& info) {
    assert(info.Depth != 0);
    uint32_t handle = 0;
    glGenTextures(1, &handle);
    if (handle == 0)
    {
        return nullptr;
    }
    uint32_t target = info.Depth == 1 ? GL_TEXTURE_2D : GL_TEXTURE_3D;
    glBindTexture(target, handle);

    if (!info.DebugName.empty()) {
        // to be able to read it in RenderDoc/errors
        glObjectLabel(GL_TEXTURE, handle, -1, info.DebugName.data());
    }

    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_REPEAT);

    auto format = TextureFormatLookUpTable[static_cast<uint32_t>(info.DataFormat)];

    if (target == GL_TEXTURE_2D)
    {
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     format.internal_format,
                     info.Width,
                     info.Height,
                     0,
                     format.format,
                     format.type,
                     nullptr);
    }
    else if (target == GL_TEXTURE_3D)
    {
        glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_REPEAT);
        glTexImage3D(GL_TEXTURE_3D,
                     0,
                     format.internal_format,
                     info.Width,
                     info.Height,
                     info.Depth,
                     0,
                     format.format,
                     format.type,
                     nullptr);
    }

    return std::unique_ptr<Texture>(new Texture(handle, target, info));
}

Texture::Texture(uint32_t handle, uint32_t target, const CreateInfo& info)
    : handle_(handle)
    , target_(target)
    , info_(info)
{
}

Texture::~Texture()
{
    glDeleteTextures(1, &handle_);
}

void Texture::bind() {
    glBindTexture(target_, handle_);
}

void Texture::upload(const void* data, [[maybe_unused]] uint32_t size)
{
    auto format = TextureFormatLookUpTable[static_cast<uint32_t>(info_.DataFormat)];
    assert(size == info_.Width * info_.Height * info_.Depth * format.size);
    bind();
    if (target_ == GL_TEXTURE_2D)
    {
        glTexSubImage2D(target_,
                        0,
                        0,
                        0,
                        info_.Width,
                        info_.Height,
                        format.format,
                        format.type,
                        data);
    }
    else if (target_ == GL_TEXTURE_3D)
    {
        glTexSubImage3D(target_,
                        0,
                        0,
                        0,
                        0,
                        info_.Width,
                        info_.Height,
                        info_.Depth,
                        format.format,
                        format.type,
                        data);
    }
}
