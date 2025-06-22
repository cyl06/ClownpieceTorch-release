# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -g -Wall -Wextra

# Include directories (if any, though not strictly needed for current structure)
# INCLUDES = -I./

# Source files
SRCS = tensor.cc test_tensor.cc meta.cc tensor_pybind.cc

# Object files
OBJS = $(SRCS:.cc=.o)

# Executable name
TARGET = test_tensor

# Default target
all: $(TARGET)

# Rule to link object files into the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

# Rule to compile .cc files into .o files
%.o: %.cc tensor.h meta.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean
