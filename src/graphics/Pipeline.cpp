#include "Pipeline.h"

#include <glad/glad.h>

#include "private_impl/GraphicsPipeline.h"
#include "private_impl/ComputePipeline.h"

std::unique_ptr<Pipeline> Pipeline::create(const Pipeline::GraphicsCreateInfo& info) {
    // Bind state which would force a recompile of the shader
    uint32_t vertexShader = glCreateShader(GL_VERTEX_SHADER);
    if (vertexShader < 0) {
        return nullptr;
    }
    {
        auto src = info.VertexShaderSource.data();
        glShaderSource(vertexShader, 1, &src, nullptr);
        glCompileShader(vertexShader);
        int success;
        glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
        char infoLog[512];
        if (success == 0) {
            glGetShaderInfoLog(vertexShader, sizeof(infoLog), nullptr, infoLog);
            std::fprintf(stderr, "%s\n", infoLog);
            return nullptr;
        }
    }
    if (!info.DebugName.empty()) {
        glObjectLabel(
            GL_SHADER, vertexShader, -1,
            (info.DebugName.data() + std::string(" vertex shader")).data());
    }

    uint32_t fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    if (fragmentShader < 0) {
        glDeleteShader(vertexShader);
        return nullptr;
    }
    {
        auto src = info.FragmentShaderSource.data();
        glShaderSource(fragmentShader, 1, &src, nullptr);
        glCompileShader(fragmentShader);
        int success;
        glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
        char infoLog[512];
        if (success == 0) {
            glGetShaderInfoLog(fragmentShader, sizeof(infoLog), nullptr,
                               infoLog);
            std::fprintf(stderr, "%s\n", infoLog);
            return nullptr;
        }
    }
    if (!info.DebugName.empty()) {
        glObjectLabel(
            GL_SHADER, fragmentShader, -1,
            (info.DebugName.data() + std::string(" fragment shader")).data());
    }

    // link the different shaders to one program (the program you see in
    // RenderDoc)
    uint32_t program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    {
        char infoLog[512];
        int success;
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(program, 512, nullptr, infoLog);
            std::fprintf(stderr, "%s\n", infoLog);
            return nullptr;
        }
    }
    glDeleteShader(fragmentShader);
    glDeleteShader(vertexShader);

    if (!info.DebugName.empty()) {
        glObjectLabel(GL_PROGRAM, program, -1, info.DebugName.data());
    }

    return std::unique_ptr<Pipeline>(
        new GraphicsPipeline(program, info.ViewportWidth, info.ViewportHeight));
}

std::unique_ptr<Pipeline> Pipeline::create(const Pipeline::ComputeCreateInfo& info) {
    uint32_t shader = glCreateShader(GL_COMPUTE_SHADER);
    if (shader < 0) {
        return nullptr;
    }
    {
        auto src = info.ShaderSource.data();
        glShaderSource(shader, 1, &src, nullptr);
        glCompileShader(shader);
        int success;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        char infoLog[512];
        if (success == 0) {
            glGetShaderInfoLog(shader, sizeof(infoLog), nullptr, infoLog);
            std::fprintf(stderr, "%s\n", infoLog);
            return nullptr;
        }
    }
    if (!info.DebugName.empty()) {
        glObjectLabel(
            GL_SHADER, shader, -1,
            (info.DebugName.data() + std::string(" compute shader")).data());
    }

    // link the different shaders to one program (the program you see in
    // RenderDoc)
    uint32_t program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);

    {
        char infoLog[512];
        int success;
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(program, 512, nullptr, infoLog);
            std::fprintf(stderr, "%s\n", infoLog);
            return nullptr;
        }
    }
    glDeleteShader(shader);

    if (!info.DebugName.empty()) {
        glObjectLabel(GL_PROGRAM, program, -1, info.DebugName.data());
    }

    return std::unique_ptr<Pipeline>(new ComputePipeline(program));
}
