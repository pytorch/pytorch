#if defined(_MSC_VER)
#include <intrin.h>
inline bool __builtin_cpu_supports(const char * avx2)
{
	bool avx2Supported = false;
	int cpuInfo[4], cpuInfo_[4];
	__cpuid(cpuInfo, 1);
	__cpuid(cpuInfo_, 0x00000007);
	bool osUsesXSAVE_XRSTORE = cpuInfo[2] & (1 << 27) || false;
	bool cpuAVXSupport = cpuInfo[2] & (1 << 28) || false;
	bool cpuAVX2Support = cpuInfo_[1] & (1 << 5) || false;
	if (osUsesXSAVE_XRSTORE && cpuAVXSupport && cpuAVX2Support)
	{
		unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
		avx2Supported = (xcrFeatureMask & 0x6) == 0x6;
	}
	return avx2Supported;
}
#endif
