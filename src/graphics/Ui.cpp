#include "Ui.h"

#include <string>
#include <glm/vec2.hpp>

#include <imgui.h>
#include <imgui_internal.h>
#include <examples/imgui_impl_sdl.h>
#include <examples/imgui_impl_opengl3.h>

void ImGuiDestroyer::operator()(ImGuiContext* context) const {
    ImGui::DestroyContext(context);
}

std::unique_ptr<Ui> Ui::create(SDL_Window* window) {
    if (!IMGUI_CHECKVERSION()) {
        return nullptr;
    }
    std::unique_ptr<ImGuiContext, ImGuiDestroyer> context;
    {
        auto ctx = ImGui::CreateContext();
        if (ctx == nullptr) {
            return nullptr;
        }
        context.reset(ctx);
    }

    // Setup Platform/Renderer bindings
    if (!ImGui_ImplSDL2_InitForOpenGL(window, nullptr)) {
        return nullptr;
    }
    if (!ImGui_ImplOpenGL3_Init()) {
        return nullptr;
    }

    return std::unique_ptr<Ui>(new Ui(std::move(context)));
}

Ui::Ui(std::unique_ptr<ImGuiContext, ImGuiDestroyer>&& context)
    : context_(std::move(context))
{
}

Ui::~Ui() = default;

void Ui::processEvent(const SDL_Event& event) {
    ImGui_ImplSDL2_ProcessEvent(&event);
}

void Ui::draw(SDL_Window *window, const std::vector<std::pair<glm::vec2, std::string>>& labels)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame(window);
    ImGui::NewFrame();

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.0f, 0.0f, 0.9f));

    for (auto& [point, str] : labels) {
        ImGui::SetNextWindowPos(ImVec2(point.x, point.y));
        ImGui::Begin(str.c_str(), nullptr, ImGuiWindowFlags_NoBackground |
                                           ImGuiWindowFlags_NoDecoration |
                                           ImGuiWindowFlags_NoInputs);
        ImGui::Text("%s", str.c_str());
        ImGui::End();
    }

    ImGui::PopStyleColor();


    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
