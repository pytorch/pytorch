#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/FusedRNNKernel.c"
#else

void THNN_(FusedRNNAssertSizes)(int factor, int count, ...)
{
  va_list list;
  va_start(list, count);

  THTensor *input = va_arg(list, THTensor*);
  THTensor *hidden = va_arg(list, THTensor*);
  THArgCheck(THTensor_(nElement)(input) == THTensor_(nElement)(hidden),
             3, "Input and Hidden tensor sizes should be the same.");

  for (int arg = 2; arg < count; ++arg){
    THTensor *tens = va_arg(list, THTensor*);
    THArgCheck(THTensor_(nElement)(input) == THTensor_(nElement)(tens)*factor,
               3, "A pointwise tensor was not the right size, should have 1/%u the elements of input/hidden tensor.", arg, factor);
  }

  va_end(list);
}

template <typename In, typename Out>
inline Out THNN_(ScalarConvert)(In v) { return static_cast<Out>(v); }

#define H2F(input) THNN_(ScalarConvert)<real, accreal>(input)
#define F2H(input) THNN_(ScalarConvert)<accreal, real>(input)

void THNN_(GRUFused_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *hidden,
          THTensor *bias1,
          THTensor *bias2,
          THTensor *hx,
          THTensor *hy,
          THTensor *storage)
{
  THTensor_(resizeAs)(hy, hx);
  THNN_(FusedRNNAssertSizes)(3, 4, input, hidden, hx, hy);
  THArgCheck(THTensor_(nElement)(storage) == THTensor_(nElement)(hx)*5,
             3, "Storage tensor for fused kernel was not sized correctly.");

  bool has_bias = (bias1!=NULL);

  size_t hsz = hx->size[1];
  size_t count = THTensor_(nElement)(hy);

  if(has_bias){
    THAssertMsg(hsz*3 == static_cast<size_t>(THTensor_(nElement)(bias1)) &&
                hsz*3 == static_cast<size_t>(THTensor_(nElement)(bias2)),
                "Bias in pointwise operation is an incorrect size, must be 3 x feature size.");
  }

  real *input_data = THTensor_(data)(input);
  real *hidden_data = THTensor_(data)(hidden);
  real *bias1_data = has_bias ? THTensor_(data)(bias1) : NULL;
  real *bias2_data = has_bias ? THTensor_(data)(bias2) : NULL;
  real *hx_data = THTensor_(data)(hx);
  real *hy_data = THTensor_(data)(hy);
  real *storage_data = THTensor_(data)(storage);

  #pragma omp parallel for
  for (size_t index = 0; index < count; ++index) {
    size_t offset = (index/hsz)*3*hsz+index%hsz;
    size_t offset_s = (index/hsz)*5*hsz+index%hsz;

    real ir = input_data[offset+0*hsz];
    real ii = input_data[offset+1*hsz];
    real in = input_data[offset+2*hsz];
    real hr = hidden_data[offset+0*hsz];
    real hi = hidden_data[offset+1*hsz];
    real hn = hidden_data[offset+2*hsz];

    real hx_ = hx_data[index];
    real *hy_ = &hy_data[index];

    real b1r, b1i, b1n, b2r, b2i, b2n;

    if (has_bias) {
      b1r = bias1_data[index%hsz+0*hsz];
      b1i = bias1_data[index%hsz+1*hsz];
      b1n = bias1_data[index%hsz+2*hsz];
      b2r = bias2_data[index%hsz+0*hsz];
      b2i = bias2_data[index%hsz+1*hsz];
      b2n = bias2_data[index%hsz+2*hsz];
    } else {
#ifndef TH_REAL_IS_HALF
      b1r = 0.0; b1i = 0.0; b1n = 0.0;
      b2r = 0.0; b2i = 0.0; b2n = 0.0;
#else
      b1r = F2H(0.0); b1i = F2H(0.0); b1n = F2H(0.0);
      b2r = F2H(0.0); b2i = F2H(0.0); b2n = F2H(0.0);
#endif
    }

    accreal rg, ig, ng;
    rg = H2F(ir) + H2F(hr) + H2F(b1r) + H2F(b2r);
    ig = H2F(ii) + H2F(hi) + H2F(b1i) + H2F(b2i);
    rg = TH_sigmoid(rg);
    ig = TH_sigmoid(ig);
    ng = H2F(in) + H2F(b1n) + rg*(H2F(hn)+H2F(b2n));
    ng = tanh(ng);

    *hy_ = F2H(ng + ig * (H2F(hx_)-ng));

    //SAVE FOR BACKWARDS
    storage_data[offset_s+0*hsz] = F2H(rg);
    storage_data[offset_s+1*hsz] = F2H(ig);
    storage_data[offset_s+2*hsz] = F2H(ng);
    storage_data[offset_s+3*hsz] = hx_;
    storage_data[offset_s+4*hsz] = F2H(H2F(hn) + H2F(b2n));
  }
}

void THNN_(GRUFused_updateGradInput)(
          THNNState *state,
          THTensor *gradInInput,
          THTensor *gradInHidden,
          THTensor *gradOutput,
          THTensor *gradInputHx,
          THTensor *storage)
{
  THTensor_(resizeAs)(gradInputHx, gradOutput);
  THNN_(FusedRNNAssertSizes)(3, 4, gradInInput, gradInHidden, gradOutput, gradInputHx);
  THArgCheck(THTensor_(nElement)(storage) == THTensor_(nElement)(gradOutput)*5,
             3, "Storage tensor for fused kernel was not sized correctly.");

  size_t hsz = gradOutput->size[1];
  size_t count = THTensor_(nElement)(gradInputHx);

  real *gradInInput_data = THTensor_(data)(gradInInput);
  real *gradInHidden_data = THTensor_(data)(gradInHidden);
  real *gradOutput_data = THTensor_(data)(gradOutput);
  real *gradInputHx_data = THTensor_(data)(gradInputHx);
  real *storage_data = THTensor_(data)(storage);

  #pragma omp parallel for
  for (size_t index = 0; index < count; ++index) {
    size_t offset = (index/hsz)*3*hsz+index%hsz;
    size_t offset_s = (index/hsz)*5*hsz+index%hsz;

    real rg = storage_data[offset_s+0*hsz];
    real ig = storage_data[offset_s+1*hsz];
    real ng = storage_data[offset_s+2*hsz];
    real hx = storage_data[offset_s+3*hsz];
    real hn = storage_data[offset_s+4*hsz];

    real go = gradOutput_data[index];

    accreal gig = H2F(go)*(H2F(hx)-H2F(ng))*(1-H2F(ig))*H2F(ig);
    accreal ghx = H2F(go)*H2F(ig);
    accreal gin = H2F(go)*(1-H2F(ig))*(1-H2F(ng)*H2F(ng));
    accreal ghn = gin * H2F(rg);
    accreal grg = gin *H2F(hn)*(1-H2F(rg))*H2F(rg);

    gradInInput_data[offset+0*hsz] = F2H(grg);
    gradInInput_data[offset+1*hsz] = F2H(gig);
    gradInInput_data[offset+2*hsz] = F2H(gin);

    gradInHidden_data[offset+0*hsz] = F2H(grg);
    gradInHidden_data[offset+1*hsz] = F2H(gig);
    gradInHidden_data[offset+2*hsz] = F2H(ghn);
    gradInputHx_data[index] = F2H(ghx);
  }
}

void THNN_(LSTMFused_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *hidden,
          THTensor *bias1,
          THTensor *bias2,
          THTensor *cx,
          THTensor *hy,
          THTensor *cy)
{
  THTensor_(resizeAs)(hy, cx);
  THTensor_(resizeAs)(cy, cx);
  THNN_(FusedRNNAssertSizes)(4, 5, input, hidden, hy, cy, cx);

  bool has_bias = (bias1!=NULL);

  size_t hsz = cx->size[1];
  size_t count = THTensor_(nElement)(hy);

  if(has_bias){
    THAssertMsg(hsz*4 == static_cast<size_t>(THTensor_(nElement)(bias1)) &&
                hsz*4 == static_cast<size_t>(THTensor_(nElement)(bias2)),
                "Bias in pointwise operation is an incorrect size, must be 4 x feature size.");
  }

  real *input_data = THTensor_(data)(input);
  real *hidden_data = THTensor_(data)(hidden);
  real *bias1_data = has_bias ? THTensor_(data)(bias1) : NULL;
  real *bias2_data = has_bias ? THTensor_(data)(bias2) : NULL;
  real *cx_data = THTensor_(data)(cx);
  real *hy_data = THTensor_(data)(hy);
  real *cy_data = THTensor_(data)(cy);

  #pragma omp parallel for
  for (size_t index = 0; index < count; ++index) {
    size_t offset = (index/hsz)*4*hsz+index%hsz;

    real *iig = &input_data[offset+0*hsz];
    real *ifg = &input_data[offset+1*hsz];
    real *icg = &input_data[offset+2*hsz];
    real *iog = &input_data[offset+3*hsz];

    real hig = hidden_data[offset+0*hsz];
    real hfg = hidden_data[offset+1*hsz];
    real hcg = hidden_data[offset+2*hsz];
    real hog = hidden_data[offset+3*hsz];

    real cx_ = cx_data[index];

    real *hy_ = &hy_data[index];
    real *cy_ = &cy_data[index];

    real b1i, b1f, b1c, b1o;
    real b2i, b2f, b2c, b2o;

    if (has_bias) {
      b1i = bias1_data[index%hsz+0*hsz];
      b1f = bias1_data[index%hsz+1*hsz];
      b1c = bias1_data[index%hsz+2*hsz];
      b1o = bias1_data[index%hsz+3*hsz];

      b2i = bias2_data[index%hsz+0*hsz];
      b2f = bias2_data[index%hsz+1*hsz];
      b2c = bias2_data[index%hsz+2*hsz];
      b2o = bias2_data[index%hsz+3*hsz];
    } else {
#ifndef TH_REAL_IS_HALF
      b1i = 0.0; b1f = 0.0; b1c = 0.0; b1o = 0.0;
      b2i = 0.0; b2f = 0.0; b2c = 0.0; b2o = 0.0;
#else
      b1i = F2H(0.0); b1f = F2H(0.0); b1c = F2H(0.0); b1o = F2H(0.0);
      b2i = F2H(0.0); b2f = F2H(0.0); b2c = F2H(0.0); b2o = F2H(0.0);
#endif
    }

    accreal ig, fg, cg, og;
    accreal f_hy, f_cy;

    ig = H2F(*iig) + H2F(hig) + H2F(b1i) + H2F(b2i);
    fg = H2F(*ifg) + H2F(hfg) + H2F(b1f) + H2F(b2f);
    cg = H2F(*icg) + H2F(hcg) + H2F(b1c) + H2F(b2c);
    og = H2F(*iog) + H2F(hog) + H2F(b1o) + H2F(b2o);

    ig = TH_sigmoid(ig);
    fg = TH_sigmoid(fg);
    cg = tanh(cg);
    og = TH_sigmoid(og);

    f_cy = (fg * H2F(cx_) ) + (ig * cg);
    f_hy = og * tanh(f_cy);

    *hy_ = F2H(f_hy);
    *cy_ = F2H(f_cy);

    //SAVE FOR BACKWARDS
    *iig = F2H(ig);
    *ifg = F2H(fg);
    *icg = F2H(cg);
    *iog = F2H(og);
  }
}

void THNN_(LSTMFused_updateGradInput)(
          THNNState *state,
          THTensor *storage,
          THTensor *gradInGates,
          THTensor *prevC,
          THTensor *cy,
          THTensor *gradOutput,
          THTensor *gradOutputCell,
          THTensor *gradInputCx)
{
  THTensor_(resizeAs)(gradInputCx, gradOutput);
  THNN_(FusedRNNAssertSizes)(4, 7, storage, gradInGates, prevC, cy,
                             gradOutput, gradOutputCell, gradInputCx);
  THArgCheck(THTensor_(nElement)(storage) == THTensor_(nElement)(gradOutput)*4,
             3, "Storage tensor for fused kernel was not sized correctly.");

  gradOutput = THTensor_(newContiguous)(gradOutput);
  gradOutputCell = THTensor_(newContiguous)(gradOutputCell);

  size_t hsz = gradOutput->size[1];
  size_t count = THTensor_(nElement)(gradInputCx);

  real *storage_data = THTensor_(data)(storage);
  real *gradInGates_data = THTensor_(data)(gradInGates);
  real *prevC_data = THTensor_(data)(prevC);
  real *cy_data = THTensor_(data)(cy);
  real *gradOutput_data = THTensor_(data)(gradOutput);
  real *gradOutputCell_data = THTensor_(data)(gradOutputCell);
  real *gradInputCx_data = THTensor_(data)(gradInputCx);

  #pragma omp parallel for
  for (size_t index = 0; index < count; ++index) {
    size_t offset = (index/hsz)*4*hsz+index%hsz;

    real ig = storage_data[offset+0*hsz];
    real fg = storage_data[offset+1*hsz];
    real cg = storage_data[offset+2*hsz];
    real og = storage_data[offset+3*hsz];

    real *ih = &gradInGates_data[offset+0*hsz];
    real *fh = &gradInGates_data[offset+1*hsz];
    real *ch = &gradInGates_data[offset+2*hsz];
    real *oh = &gradInGates_data[offset+3*hsz];

    real cx_ = prevC_data[index];
    real cy_ = cy_data[index];

    real *gi = &gradInputCx_data[index];

    real go = gradOutput_data[index];
    real goc = gradOutputCell_data[index];

    accreal gcx = tanh(H2F(cy_));

    accreal gog = H2F(go) * gcx;
    gcx = H2F(go) * H2F(og) * (1 - gcx*gcx) + H2F(goc);

    accreal gig = gcx * H2F(cg);
    accreal gfg = gcx * H2F(cx_);
    accreal gcg = gcx * H2F(ig);

    gcx = gcx * H2F(fg);

    gig = gig * (1-H2F(ig)) * H2F(ig);
    gfg = gfg * (1-H2F(fg)) * H2F(fg);
    gcg = gcg * (1-H2F(cg)*H2F(cg));
    gog = gog * (1-H2F(og)) * H2F(og);

    *ih = F2H(gig);
    *fh = F2H(gfg);
    *ch = F2H(gcg);
    *oh = F2H(gog);

    *gi = F2H(gcx);
  }
}

#endif
