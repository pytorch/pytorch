#!/bin/bash
# Script used only in CD pipeline

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
index a5007ffc..13fa07fc 100644
--- a/amdgpu/amdgpu_asic_id.c
+++ b/amdgpu/amdgpu_asic_id.c
@@ -22,6 +22,13 @@
  *
  */

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
@@ -34,6 +41,19 @@
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
@@ -113,10 +133,46 @@ void amdgpu_parse_asic_ids(struct amdgpu_device *dev)
 	int line_num = 1;
 	int r = 0;

+	// attempt to find typical location for amdgpu.ids file
 	fp = fopen(AMDGPU_ASIC_ID_TABLE, "r");
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
+		amdgpuids_path_msg = AMDGPU_ASIC_ID_TABLE;
+	}
+
+	// both hard-coded location and search have failed
 	if (!fp) {
-		fprintf(stderr, "%s: %s\n", AMDGPU_ASIC_ID_TABLE,
-			strerror(errno));
+		fprintf(stderr, "amdgpu.ids: No such file or directory\n");
 		return;
 	}

@@ -132,7 +188,7 @@ void amdgpu_parse_asic_ids(struct amdgpu_device *dev)
 			continue;
 		}

-		drmMsg("%s version: %s\n", AMDGPU_ASIC_ID_TABLE, line);
+		drmMsg("%s version: %s\n", amdgpuids_path_msg, line);
 		break;
 	}

@@ -150,7 +206,7 @@ void amdgpu_parse_asic_ids(struct amdgpu_device *dev)

 	if (r == -EINVAL) {
 		fprintf(stderr, "Invalid format: %s: line %d: %s\n",
-			AMDGPU_ASIC_ID_TABLE, line_num, line);
+			amdgpuids_path_msg, line_num, line);
 	} else if (r && r != -EAGAIN) {
 		fprintf(stderr, "%s: Cannot parse ASIC IDs: %s\n",
 			__func__, strerror(-r));
EOF

###########################
### build
###########################
meson builddir --prefix=/opt/amdgpu
pushd builddir
ninja install

popd
popd
