
static void tensorGesv(rpc::RPCMessage& raw_message) {
  at::Tensor rb = unpackRetrieveTensor(raw_message);
  at::Tensor ra = unpackRetrieveTensor(raw_message);
  at::Tensor b = unpackRetrieveTensor(raw_message);
  at::Tensor a = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::gesv_out(rb, ra, b, a);
}

static void tensorTrtrs(rpc::RPCMessage& raw_message) {
  at::Tensor rb = unpackRetrieveTensor(raw_message);
  at::Tensor ra = unpackRetrieveTensor(raw_message);
  at::Tensor b = unpackRetrieveTensor(raw_message);
  at::Tensor a = unpackRetrieveTensor(raw_message);
  auto uplo = unpackInteger(raw_message);
  auto trans = unpackInteger(raw_message);
  auto diag = unpackInteger(raw_message);
  at::trtrs_out(rb, ra, b, a, uplo, trans, diag);
}

static void tensorGels(rpc::RPCMessage& raw_message) {
  at::Tensor rb = unpackRetrieveTensor(raw_message);
  at::Tensor ra = unpackRetrieveTensor(raw_message);
  at::Tensor b = unpackRetrieveTensor(raw_message);
  at::Tensor a = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::gels_out(rb, ra, b, a);
}

static void tensorSyev(rpc::RPCMessage& raw_message) {
  at::Tensor re = unpackRetrieveTensor(raw_message);
  at::Tensor rv = unpackRetrieveTensor(raw_message);
  at::Tensor a = unpackRetrieveTensor(raw_message);
  auto jobz = unpackInteger(raw_message);
  auto uplo = unpackInteger(raw_message);
  finalize(raw_message);
  at::symeig_out(re, rv, a, jobz, uplo);
}

static void tensorGeev(rpc::RPCMessage& raw_message) {
  at::Tensor re = unpackRetrieveTensor(raw_message);
  at::Tensor rv = unpackRetrieveTensor(raw_message);
  at::Tensor a = unpackRetrieveTensor(raw_message);
  auto jobvr = unpackInteger(raw_message);
  finalize(raw_message);
  at::eig_out(re, rv, a, jobvr);
}

static void tensorGesvd2(rpc::RPCMessage& raw_message) {
  at::Tensor ru = unpackRetrieveTensor(raw_message);
  at::Tensor rs = unpackRetrieveTensor(raw_message);
  at::Tensor rv = unpackRetrieveTensor(raw_message);
  at::Tensor ra = unpackRetrieveTensor(raw_message);
  at::Tensor a = unpackRetrieveTensor(raw_message);
  auto jobu = unpackInteger(raw_message);
  finalize(raw_message);
  throw std::runtime_error("gesv2d not implemented in ATen");
  /* ru->gesvd2(*rs, *rv, *ra, *a, &jobu); */
}

static void tensorGetri(rpc::RPCMessage& raw_message) {
  at::Tensor ra = unpackRetrieveTensor(raw_message);
  at::Tensor a = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::inverse_out(ra, a);
}

static void tensorPotrf(rpc::RPCMessage& raw_message) {
  at::Tensor ra = unpackRetrieveTensor(raw_message);
  at::Tensor a = unpackRetrieveTensor(raw_message);
  auto uplo = unpackInteger(raw_message);
  finalize(raw_message);
  at::potrf_out(ra, a, uplo);
}

static void tensorPotrs(rpc::RPCMessage& raw_message) {
  at::Tensor rb = unpackRetrieveTensor(raw_message);
  at::Tensor b = unpackRetrieveTensor(raw_message);
  at::Tensor a = unpackRetrieveTensor(raw_message);
  auto uplo = unpackInteger(raw_message);
  finalize(raw_message);
  at::potrs_out(rb, b, a, uplo);
}

static void tensorPotri(rpc::RPCMessage& raw_message) {
  at::Tensor ra = unpackRetrieveTensor(raw_message);
  at::Tensor a = unpackRetrieveTensor(raw_message);
  auto uplo = unpackInteger(raw_message);
  finalize(raw_message);
  at::potri_out(ra, a, uplo);
}

static void tensorQr(rpc::RPCMessage& raw_message) {
  at::Tensor rq = unpackRetrieveTensor(raw_message);
  at::Tensor rr = unpackRetrieveTensor(raw_message);
  at::Tensor a = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::qr_out(rq, rr, a);
}

static void tensorGeqrf(rpc::RPCMessage& raw_message) {
  at::Tensor ra = unpackRetrieveTensor(raw_message);
  at::Tensor rtau = unpackRetrieveTensor(raw_message);
  at::Tensor a = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::geqrf_out(ra, rtau, a);
}

static void tensorOrgqr(rpc::RPCMessage& raw_message) {
  at::Tensor ra = unpackRetrieveTensor(raw_message);
  at::Tensor a = unpackRetrieveTensor(raw_message);
  at::Tensor tau = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  at::orgqr_out(ra, a, tau);
}

static void tensorOrmqr(rpc::RPCMessage& raw_message) {
  at::Tensor ra = unpackRetrieveTensor(raw_message);
  at::Tensor a = unpackRetrieveTensor(raw_message);
  at::Tensor tau = unpackRetrieveTensor(raw_message);
  at::Tensor c = unpackRetrieveTensor(raw_message);
  auto side = unpackInteger(raw_message);
  auto trans = unpackInteger(raw_message);
  finalize(raw_message);
  at::ormqr_out(ra, ra, tau, c, side, trans);
}

static void tensorPstrf(rpc::RPCMessage& raw_message) {
  at::Tensor ra = unpackRetrieveTensor(raw_message);
  at::Tensor rpiv = unpackRetrieveTensor(raw_message);
  at::Tensor a = unpackRetrieveTensor(raw_message);
  auto uplo = unpackInteger(raw_message);
  RPCType type = peekType(raw_message);
  if (isInteger(type)) {
    auto tol = unpackInteger(raw_message);
    finalize(raw_message);
    at::pstrf_out(ra, rpiv, a, uplo, tol);
  } else if (isFloat(type)) {
    auto tol = unpackFloat(raw_message);
    finalize(raw_message);
    at::pstrf_out(ra, rpiv, a, uplo, tol);
  } else {
    throw std::runtime_error("expected scalar type");
  }
}
