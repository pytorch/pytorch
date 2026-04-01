// multi-file-defines.h

#ifdef __SLANG__
#define R(X) /**/
#define BEGIN_CBUFFER(NAME) cbuffer NAME
#define END_CBUFFER(NAME, REG) /**/
#define CBUFFER_REF(NAME, FIELD) FIELD
#define PUBLIC public
#else
#define R(X) /*X*/
#define BEGIN_CBUFFER(NAME) struct SLANG_ParameterGroup_##NAME
#define END_CBUFFER(NAME, REG)            \
    ;                                     \
    cbuffer NAME /*REG*/                  \
    {                                     \
        SLANG_ParameterGroup_##NAME NAME; \
    }
#define CBUFFER_REF(NAME, FIELD) NAME.FIELD
#define PUBLIC
#define sharedC sharedC_0
#define sharedCA sharedCA_0
#define sharedCB sharedCB_0
#define sharedCC sharedCC_0
#define sharedCD sharedCD_0

#define vertexC vertexC_0
#define vertexCA vertexCA_0
#define vertexCB vertexCB_0
#define vertexCC vertexCC_0
#define vertexCD vertexCD_0

#define fragmentC fragmentC_0
#define fragmentCA fragmentCA_0
#define fragmentCB fragmentCB_0
#define fragmentCC fragmentCC_0
#define fragmentCD fragmentCD_0

#define sharedS sharedS_0
#define sharedT sharedT_0
#define sharedTV sharedTV_0
#define sharedTF sharedTF_0

#define vertexS vertexS_0
#define vertexT vertexT_0

#define fragmentS fragmentS_0
#define fragmentT fragmentT_0

#endif
