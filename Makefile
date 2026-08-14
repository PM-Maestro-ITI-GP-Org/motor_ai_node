# motor_ai_node links nothing outside this tree -- no CommonAPI, no vsomeip, no
# boost, no TensorFlow, no scikit-learn -- so there is no configure step, no
# toolchain file and no dependency on the SOME/IP tree that motor_ai_server
# needs. One compiler, and sources that are all in this repository.
#
# What it does need, and why:
#
#   include/      the feature code, the three models, and the JSON parser, all
#                 header-only. Kept here rather than shared with the training
#                 repository so a Yocto build has one thing to fetch.
#   kissfft/      the FFT, vendored, built in DOUBLE precision. numpy computes
#                 in float64 and the difference showed up as ~1e-6 drift in
#                 every FFT-derived feature -- exactly the size of the parity
#                 tolerance the two implementations are checked against.
#   model/,
#   config/       the trained artefacts, read at startup. Not compiled in: they
#                 are data, and replacing a model must not need a rebuild.
#
# C++17 is required by include/ (structured bindings). The node's own source is
# still plain C++ and would build at C++14.
#
# CXX and CXXFLAGS are ?= so the environment or the command line can replace
# them, which is how Yocto cross-compiles this:
#
#     make CXX="$CXX" CXXFLAGS="$CXXFLAGS -std=c++17" LDFLAGS="$LDFLAGS"
#
# LDFLAGS is in the rule because the compile and the link are one invocation;
# OE's QA check fails a package whose binaries were linked without them.
CXX      ?= g++
CC       ?= gcc
CXXFLAGS ?= -O2 -Wall -Wextra -std=c++17
CFLAGS   ?= -O2 -Wall

# kissfft defaults to single precision; see above.
CPPFLAGS += -Iinclude -Ikissfft -Dkiss_fft_scalar=double

BIN      := motor_ai_node
SRCS     := src/motor_ai_node.cpp
CSRCS    := kissfft/kiss_fft.c kissfft/kiss_fftr.c
COBJS    := $(CSRCS:.c=.o)

.PHONY: all clean

all: $(BIN)

$(BIN): $(SRCS) $(COBJS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -o $@ $(SRCS) $(COBJS)

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(BIN) $(COBJS)
