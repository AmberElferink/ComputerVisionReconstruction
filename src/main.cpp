#include <cstdlib>
#include <filesystem>

#include "VoxelReconstruction.h"

using namespace nl_uu_science_gmt;

int main(
		int argc, char** argv)
{
	VoxelReconstruction::showKeys();
	VoxelReconstruction vr(std::filesystem::path("data"), 4);
	vr.run(argc, argv);

	return EXIT_SUCCESS;
}
