FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG PLANARGS_PATH_ARG=/workspace
ENV PLANARGS_PATH=${PLANARGS_PATH_ARG}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/root/.local/bin:$PATH

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf autoconf-archive automake bison flex gperf m4 meson ninja-build \
    build-essential pkg-config \
    git curl unzip zip tar \
    python3 python3-dev python3-pip python3-venv \
    cmake ffmpeg \
    \
    # X11 / EGL / Vulkan tools (GUI debug + headless)
    xauth x11-utils \
    libdbus-1-3 libglib2.0-0 libsm6 libfontconfig1 libfreetype6 \
    libx11-6 libx11-dev libxext6 libxext-dev libxrender1 libxrender-dev \
    libxcb1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render0 \
    libxcb-render-util0 libxcb-shape0 libxcb-shm0 libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 \
    libxkbcommon0 libxkbcommon-x11-0 libxkbfile1 libxmu6 libxaw7 libxxf86dga1 \
    libgl1 libgl1-mesa-dev libopengl0 libglu1-mesa-dev freeglut3-dev \
    libegl1 libglew-dev libgl1-mesa-dev libglu1-mesa-dev \
    vulkan-tools zenity \
    \
    # COLMAP / OpenMVS deps
    libeigen3-dev \
    libboost-filesystem-dev libboost-graph-dev libboost-iostreams-dev \
    libboost-program-options-dev libboost-serialization-dev libboost-system-dev \
    libsuitesparse-dev libmetis-dev libtbb-dev \
    libceres-dev libgoogle-glog-dev libgflags-dev \
    libjpeg-dev libpng-dev libtiff-dev libfreeimage-dev \
    libopencv-dev \
    libsqlite3-dev sqlite3 \
    libcgal-dev libcgal-qt5-dev \
    libflann-dev libnanoflann-dev \
    libjxl-dev libjxl0.7 \
    qtbase5-dev libqt5opengl5-dev \
    \
    && rm -rf /var/lib/apt/lists/*

# Runtime dir for XDG
RUN mkdir -p /tmp/runtime-root && chmod 1777 /tmp/runtime-root

# Build and install OpenMVS (patched for Ubuntu 24.04 toolchain)
RUN apt-get update; \
    git clone https://github.com/cdcseacave/VCG.git /tmp/vcglib; \
    git clone --recurse-submodules https://github.com/cdcseacave/openMVS.git /tmp/openMVS; \
    sed -i 's/IMWRITE_JPEGXL_QUALITY/IMWRITE_JPEG_QUALITY/g' /tmp/openMVS/libs/Common/Types.inl; \
    sed -i 's#CGAL/AABB_traits_3.h#CGAL/AABB_traits.h#g' /tmp/openMVS/libs/MVS/SceneReconstruct.cpp; \
    sed -i 's/CGAL::AABB_traits_3/CGAL::AABB_traits/g' /tmp/openMVS/libs/MVS/SceneReconstruct.cpp; \
    sed -i 's#CGAL/AABB_triangle_primitive_3.h#CGAL/AABB_triangle_primitive.h#g' /tmp/openMVS/libs/MVS/SceneReconstruct.cpp; \
    cd /tmp/openMVS; \
    mkdir -p make; \
    cd make; \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DVCG_ROOT="/tmp/vcglib" -DOpenMVS_USE_CUDA=OFF -DPython3_EXECUTABLE=/usr/bin/python3 -DBoost_NO_BOOST_CMAKE=ON -DCMAKE_POLICY_DEFAULT_CMP0146=OLD; \
    make -j"$(nproc)"; \
    make install; \
    rm -rf /tmp/openMVS /tmp/vcglib

# Build and Install COLMAP
RUN git clone https://github.com/colmap/colmap.git -b 3.9 /tmp/colmap; \
    cd /tmp/colmap; \
    mkdir build; \
    cd build; \
    cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90"; \
    ninja; \
    ninja install; \
    rm -rf /tmp/colmap

# Set PATHs for OpenMVS and COLMAP
ENV PATH="/usr/local/bin/OpenMVS:/usr/local/bin/colmap:${PATH}"
RUN echo 'export PATH="/usr/local/bin/OpenMVS:/usr/local/bin/colmap:${PATH}"' >> /root/.bashrc

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR ${PLANARGS_PATH}

CMD ["/bin/bash"]
