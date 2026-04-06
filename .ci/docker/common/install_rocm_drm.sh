#!/bin/bash
# Script used only in CD pipeline

PREFIX="$1"

###########################
### prereqs
###########################
# Install Python packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    apt-get update -y
    apt-get install -y libpciaccess-dev pkg-config
    apt-get clean
    ;;
  centos|almalinux)
    yum install -y libpciaccess-devel pkgconfig
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
python3 -m pip install meson ninja

###########################
### clone repo
###########################
GIT_SSL_NO_VERIFY=true git clone https://gitlab.freedesktop.org/mesa/drm.git
pushd drm

###########################
### patch
###########################
patch -p1 <<'EOF'
diff --git a/amdgpu/amdgpu_asic_id.c b/amdgpu/amdgpu_asic_id.c
index cd8ee596..3e453ecd 100644
--- a/amdgpu/amdgpu_asic_id.c
+++ b/amdgpu/amdgpu_asic_id.c
@@ -27,6 +27,13 @@
 #define _GNU_SOURCE
 #endif

+#define _XOPEN_SOURCE 700
+#define _LARGEFILE64_SOURCE
+#define _FILE_OFFSET_BITS 64
+#include <ftw.h>
+#include <link.h>
+#include <limits.h>
+
 #include <ctype.h>
 #include <stdio.h>
 #include <stdlib.h>
@@ -39,6 +46,19 @@
 #include "amdgpu_drm.h"
 #include "amdgpu_internal.h"

+static char *amdgpuids_path = NULL;
+static const char* amdgpuids_path_msg = NULL;
+
+static int check_for_location_of_amdgpuids(const char *filepath, const struct stat *info, const int typeflag, struct FTW *pathinfo)
+{
+	if (typeflag == FTW_F && strstr(filepath, "amdgpu.ids")) {
+		amdgpuids_path = strdup(filepath);
+		return 1;
+	}
+
+	return 0;
+}
+
 static int parse_one_line(struct amdgpu_device *dev, const char *line)
 {
 	char *buf, *saveptr;
@@ -290,9 +310,46 @@ void amdgpu_parse_asic_ids(struct amdgpu_device *dev)
 	if (!amdgpu_asic_id_table_path)
 		amdgpu_asic_id_table_path = strdup(AMDGPU_ASIC_ID_TABLE);

+	// attempt to find typical location for amdgpu.ids file
 	fp = fopen(amdgpu_asic_id_table_path, "r");
+
+	// if it doesn't exist, search
+	if (!fp) {
+
+	char self_path[ PATH_MAX ];
+	ssize_t count;
+	ssize_t i;
+
+	count = readlink( "/proc/self/exe", self_path, PATH_MAX );
+	if (count > 0) {
+		self_path[count] = '\0';
+
+		// remove '/bin/python' from self_path
+		for (i=count; i>0; --i) {
+			if (self_path[i] == '/') break;
+			self_path[i] = '\0';
+		}
+		self_path[i] = '\0';
+		for (; i>0; --i) {
+			if (self_path[i] == '/') break;
+			self_path[i] = '\0';
+		}
+		self_path[i] = '\0';
+
+		if (1 == nftw(self_path, check_for_location_of_amdgpuids, 5, FTW_PHYS)) {
+			fp = fopen(amdgpuids_path, "r");
+			amdgpuids_path_msg = amdgpuids_path;
+		}
+	}
+
+	}
+	else {
+		amdgpuids_path_msg = amdgpu_asic_id_table_path;
+	}
+
+	// both hard-coded location and search have failed
 	if (!fp) {
-		fprintf(stderr, "%s: %s\n", amdgpu_asic_id_table_path,
+		fprintf(stderr, "%s: %s\n", amdgpuids_path_msg,
 			strerror(errno));
 		goto get_cpu;
 	}
@@ -309,7 +366,7 @@ void amdgpu_parse_asic_ids(struct amdgpu_device *dev)
 			continue;
 		}

-		drmMsg("%s version: %s\n", amdgpu_asic_id_table_path, line);
+		drmMsg("%s version: %s\n", amdgpuids_path_msg, line);
 		break;
 	}

@@ -327,7 +384,7 @@ void amdgpu_parse_asic_ids(struct amdgpu_device *dev)

 	if (r == -EINVAL) {
 		fprintf(stderr, "Invalid format: %s: line %d: %s\n",
-			amdgpu_asic_id_table_path, line_num, line);
+			amdgpuids_path_msg, line_num, line);
 	} else if (r && r != -EAGAIN) {
 		fprintf(stderr, "%s: Cannot parse ASIC IDs: %s\n",
 			__func__, strerror(-r));
@@ -338,6 +395,7 @@ void amdgpu_parse_asic_ids(struct amdgpu_device *dev)

 get_cpu:
 	free(amdgpu_asic_id_table_path);
+	if (amdgpuids_path) free(amdgpuids_path);
 	if (dev->info.ids_flags & AMDGPU_IDS_FLAGS_FUSION &&
 	    dev->marketing_name == NULL) {
 		amdgpu_parse_proc_cpuinfo(dev);
EOF

###########################
### build
###########################
meson builddir --prefix=${PREFIX}
pushd builddir
ninja install

popd
popd
