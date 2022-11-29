cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(DGAN_module VERSION 0.1 DESCRIPTION "a ML-torch module for use as a discriminator in GANS" LANGUAGES CXX)

find_package(Torch REQUIRED)

include(FetchContent)
add_library(DGAN INTERFACE)

if(USE_YAML)
	target_compile_definitions(DGAN INTERFACE USE_YAML)
	if(NOT TARGET yaml-cpp)
		find_package(YAML-CPP CONFIG QUIET)
		if(NOT YAML-CPP_FOUND)
			message(STATUS "YAML-CPP was not found. Downloading")

			FetchContent_Declare(
				yaml-cpp
				GIT_REPOSITORY git@github.com:jbeder/yaml-cpp.git
				GIT_TAG yaml-cpp-0.7.0
				)
			FetchContent_MakeAvailable(yaml-cpp)
			install(TARGETS yaml-cpp EXPORT DGANTargets DESTINATION lib)
		else()
			message(STATUS "YAML FOUND in ${YAML-CPP_DIR}")
		endif(NOT YAML-CPP_FOUND)
	endif(NOT TARGET yaml-cpp)
	target_link_libraries(DGAN INTERFACE  yaml-cpp)
endif(USE_YAML)


target_link_libraries(DGAN  INTERFACE "${TORCH_LIBRARIES}")

target_include_directories(DGAN INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	"$<INSTALL_INTERFACE:include>")

install(TARGETS DGAN EXPORT DGANTargets DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/)

install(EXPORT DGANTargets
	FILE DGANTargets.cmake
	DESTINATION lib/cmake/DGAN
	)
include(CMakePackageConfigHelpers)
# generate the config file that is includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}/DGANConfig.cmake"
	INSTALL_DESTINATION "lib/cmake/DGAN"
	NO_SET_AND_CHECK_MACRO
	NO_CHECK_REQUIRED_COMPONENTS_MACRO
	)
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/DGANConfigVersion.cmake"
	VERSION "0.1.1.0"
	COMPATIBILITY AnyNewerVersion
	)
install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/DGANConfig.cmake
	${CMAKE_CURRENT_BINARY_DIR}/DGANConfigVersion.cmake
	DESTINATION lib/cmake/DGAN
	)
export(EXPORT DGANTargets
	FILE "${CMAKE_CURRENT_BINARY_DIR}/DGANTargets.cmake"
	)
execute_process(COMMAND "git" "log" "--format='%H'" "-n 1"
	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
	OUTPUT_VARIABLE  commit)
execute_process(COMMAND "git" "rev-parse" "--abbrev-ref" "HEAD"
	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
	OUTPUT_VARIABLE  branch)

FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt  ${PROJECT_NAME}:\n)
FILE(APPEND ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt  \t commit=${commit})
FILE(APPEND ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt  \t branch=${branch}\n)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt DESTINATION .conf)
