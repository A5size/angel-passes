# Makefile for compiling the Fortran simulation library

# Compiler
FC = gfortran

# Source file
SRC = angel_passes.f90

# Output
TARGET = libfort.so

# Build options
FLAGS_RELEASE = -fPIC -shared -O3
FLAGS_DEBUG   = -fPIC -shared -Wall -fbounds-check -O0 -Wuninitialized -fbacktrace

# Default target (release build)
all: $(TARGET)

$(TARGET): $(SRC)
	$(FC) $(FLAGS_RELEASE) -o $(TARGET) $(SRC)

# Debug build
debug:
	$(FC) $(FLAGS_DEBUG) -o $(TARGET) $(SRC)

# Clean build artifacts
clean:
	rm -f $(TARGET)
