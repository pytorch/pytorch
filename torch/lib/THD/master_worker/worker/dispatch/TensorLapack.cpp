
static void tensorGesv(rpc::RPCMessage& raw_message) {
  thpp::Tensor *rb = unpackRetrieveTensor(raw_message);
  thpp::Tensor *ra = unpackRetrieveTensor(raw_message);
  thpp::Tensor *b = unpackRetrieveTensor(raw_message);
  thpp::Tensor *a = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  rb->gesv(*ra, *b, *a);
}

static void tensorTrtrs(rpc::RPCMessage& raw_message) {
  thpp::Tensor *rb = unpackRetrieveTensor(raw_message);
  thpp::Tensor *ra = unpackRetrieveTensor(raw_message);
  thpp::Tensor *b = unpackRetrieveTensor(raw_message);
  thpp::Tensor *a = unpackRetrieveTensor(raw_message);
  char uplo = unpackInteger(raw_message);
  char trans = unpackInteger(raw_message);
  char diag = unpackInteger(raw_message);
  rb->trtrs(*ra, *b, *a, &uplo, &trans, &diag);
}

static void tensorGels(rpc::RPCMessage& raw_message) {
  thpp::Tensor *rb = unpackRetrieveTensor(raw_message);
  thpp::Tensor *ra = unpackRetrieveTensor(raw_message);
  thpp::Tensor *b = unpackRetrieveTensor(raw_message);
  thpp::Tensor *a = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  rb->gels(*ra, *b, *a);
}

static void tensorSyev(rpc::RPCMessage& raw_message) {
  thpp::Tensor *re = unpackRetrieveTensor(raw_message);
  thpp::Tensor *rv = unpackRetrieveTensor(raw_message);
  thpp::Tensor *a = unpackRetrieveTensor(raw_message);
  char jobz = unpackInteger(raw_message);
  char uplo = unpackInteger(raw_message);
  finalize(raw_message);
  re->syev(*rv, *a, &jobz, &uplo);
}

static void tensorGeev(rpc::RPCMessage& raw_message) {
  thpp::Tensor *re = unpackRetrieveTensor(raw_message);
  thpp::Tensor *rv = unpackRetrieveTensor(raw_message);
  thpp::Tensor *a = unpackRetrieveTensor(raw_message);
  char jobvr = unpackInteger(raw_message);
  finalize(raw_message);
  re->geev(*rv, *a, &jobvr);
}

static void tensorGesvd2(rpc::RPCMessage& raw_message) {
  thpp::Tensor *ru = unpackRetrieveTensor(raw_message);
  thpp::Tensor *rs = unpackRetrieveTensor(raw_message);
  thpp::Tensor *rv = unpackRetrieveTensor(raw_message);
  thpp::Tensor *ra = unpackRetrieveTensor(raw_message);
  thpp::Tensor *a = unpackRetrieveTensor(raw_message);
  char jobu = unpackInteger(raw_message);
  finalize(raw_message);
  ru->gesvd2(*rs, *rv, *ra, *a, &jobu);
}

static void tensorGetri(rpc::RPCMessage& raw_message) {
  thpp::Tensor *ra = unpackRetrieveTensor(raw_message);
  thpp::Tensor *a = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  ra->getri(*a);
}

static void tensorPotrf(rpc::RPCMessage& raw_message) {
  thpp::Tensor *ra = unpackRetrieveTensor(raw_message);
  thpp::Tensor *a = unpackRetrieveTensor(raw_message);
  char uplo = unpackInteger(raw_message);
  finalize(raw_message);
  ra->potrf(*a, &uplo);
}

static void tensorPotrs(rpc::RPCMessage& raw_message) {
  thpp::Tensor *rb = unpackRetrieveTensor(raw_message);
  thpp::Tensor *b = unpackRetrieveTensor(raw_message);
  thpp::Tensor *a = unpackRetrieveTensor(raw_message);
  char uplo = unpackInteger(raw_message);
  finalize(raw_message);
  rb->potrs(*b, *a, &uplo);
}

static void tensorPotri(rpc::RPCMessage& raw_message) {
  thpp::Tensor *ra = unpackRetrieveTensor(raw_message);
  thpp::Tensor *a = unpackRetrieveTensor(raw_message);
  char uplo = unpackInteger(raw_message);
  finalize(raw_message);
  ra->potri(*a, &uplo);
}

static void tensorQr(rpc::RPCMessage& raw_message) {
  thpp::Tensor *rq = unpackRetrieveTensor(raw_message);
  thpp::Tensor *rr = unpackRetrieveTensor(raw_message);
  thpp::Tensor *a = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  rq->qr(*rr, *a);
}

static void tensorGeqrf(rpc::RPCMessage& raw_message) {
  thpp::Tensor *ra = unpackRetrieveTensor(raw_message);
  thpp::Tensor *rtau = unpackRetrieveTensor(raw_message);
  thpp::Tensor *a = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  ra->geqrf(*rtau, *a);
}

static void tensorOrgqr(rpc::RPCMessage& raw_message) {
  thpp::Tensor *ra = unpackRetrieveTensor(raw_message);
  thpp::Tensor *a = unpackRetrieveTensor(raw_message);
  thpp::Tensor *tau = unpackRetrieveTensor(raw_message);
  finalize(raw_message);
  ra->geqrf(*a, *tau);
}

static void tensorOrmqr(rpc::RPCMessage& raw_message) {
  thpp::Tensor *ra = unpackRetrieveTensor(raw_message);
  thpp::Tensor *a = unpackRetrieveTensor(raw_message);
  thpp::Tensor *tau = unpackRetrieveTensor(raw_message);
  thpp::Tensor *c = unpackRetrieveTensor(raw_message);
  char side = unpackInteger(raw_message);
  char trans = unpackInteger(raw_message);
  finalize(raw_message);
  ra->ormqr(*a, *tau, *c, &side, &trans);
}

static void tensorPstrf(rpc::RPCMessage& raw_message) {
  thpp::Tensor *ra = unpackRetrieveTensor(raw_message);
  thpp::Tensor *rpiv = unpackRetrieveTensor(raw_message);
  thpp::Tensor *a = unpackRetrieveTensor(raw_message);
  char uplo = unpackInteger(raw_message);
  thpp::Type type = peekType(raw_message);
  if (thpp::isInteger(type)) {
    auto tol = unpackInteger(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::IntTensor*>(ra)->pstrf(*rpiv, *a, &uplo, tol);
  } else if (thpp::isFloat(type)) {
    auto tol = unpackFloat(raw_message);
    finalize(raw_message);
    dynamic_cast<thpp::FloatTensor*>(ra)->pstrf(*rpiv, *a, &uplo, tol);
  } else {
    throw std::runtime_error("expected scalar type");
  }
}
