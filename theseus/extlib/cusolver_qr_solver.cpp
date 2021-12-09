
#include "cusolver_sp_defs.h"

#include <pybind11/pybind11.h>
#include <iostream>
#include <torch/extension.h>
#include <functional>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <torch/extension.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>
#include <ATen/cuda/CUDAContext.h>
#include <cusolverSp.h>

struct CusolverQRSolver {

    CusolverQRSolver(int64_t cols,
                     const torch::Tensor& A_rowPtr,
                     const torch::Tensor& A_colInd);

	~CusolverQRSolver();
		
	torch::Tensor factorAndSolve(int batchSize,
	                             const torch::Tensor& A_val,
	                             const torch::Tensor& b);

    int64_t cols;
    int64_t rows;
    int64_t nnz;

    torch::Tensor A_rowPtr;
    torch::Tensor A_colInd;
	
	cusparseMatDescr_t A_descr = nullptr;
	csrqrInfo_t info = nullptr;
};

CusolverQRSolver::CusolverQRSolver(int64_t cols,
                                   const torch::Tensor& A_rowPtr,
                                   const torch::Tensor& A_colInd)
	: cols(cols), A_rowPtr(A_rowPtr), A_colInd(A_colInd) {

	rows = A_rowPtr.size(0) - 1;
	nnz = A_colInd.size(0);

	TORCH_CHECK(A_rowPtr.device().is_cuda());
	TORCH_CHECK(A_colInd.device().is_cuda());
	TORCH_CHECK(A_rowPtr.dtype() == torch::kInt);
	TORCH_CHECK(A_colInd.dtype() == torch::kInt);
	TORCH_CHECK(A_rowPtr.dim() == 1);
	TORCH_CHECK(A_colInd.dim() == 1);

	const int *pA_rowPtr = A_rowPtr.data_ptr<int>();
	const int *pA_colInd = A_colInd.data_ptr<int>();
	
	cusolverSpHandle_t cuSpHandle = theseus::cusolver_sp::getCurrentCUDASolverSpHandle();
 
	TORCH_CUDASPARSE_CHECK(cusparseCreateMatDescr(&A_descr));
	TORCH_CUDASPARSE_CHECK(cusparseSetMatIndexBase(A_descr, CUSPARSE_INDEX_BASE_ZERO));
	TORCH_CUDASPARSE_CHECK(cusparseSetMatType(A_descr, CUSPARSE_MATRIX_TYPE_GENERAL));

	CUSOLVER_CHECK(cusolverSpCreateCsrqrInfo(&info));

	// symbolic analysis
	CUSOLVER_CHECK(cusolverSpXcsrqrAnalysisBatched(cuSpHandle, rows, cols, nnz,
	                                               A_descr, pA_rowPtr, pA_colInd, info));
}

CusolverQRSolver::~CusolverQRSolver() {
	CUSOLVER_CHECK(cusolverSpDestroyCsrqrInfo(info));
	TORCH_CUDASPARSE_CHECK(cusparseDestroyMatDescr(A_descr));
}

torch::Tensor CusolverQRSolver::factorAndSolve(int batchSize,
                                               const torch::Tensor& As_val,
                                               const torch::Tensor& b) {

	TORCH_CHECK(As_val.device().is_cuda());
	TORCH_CHECK(b.device().is_cuda());
	TORCH_CHECK(As_val.dtype() == torch::kDouble); // TODO: add support for float
	TORCH_CHECK(b.dtype() == torch::kDouble);
	TORCH_CHECK(As_val.dim() == 2);
	TORCH_CHECK(As_val.size(0) == batchSize);
	TORCH_CHECK(As_val.size(1) == nnz);
	TORCH_CHECK(b.size(0) == batchSize);
	TORCH_CHECK(b.size(1) == rows);

	const int *pA_rowPtr = A_rowPtr.data_ptr<int>();
	const int *pA_colInd = A_colInd.data_ptr<int>();
	const double *pA_val = As_val.data_ptr<double>();

	cusolverSpHandle_t cuSpHandle = theseus::cusolver_sp::getCurrentCUDASolverSpHandle();

	// get size for this batch_size
	size_t internalDataInBytes;
	size_t workspaceInBytes;
	CUSOLVER_CHECK(cusolverSpDcsrqrBufferInfoBatched(cuSpHandle, rows, cols, nnz,
	                                                 A_descr, pA_val, pA_rowPtr, pA_colInd,
	                                                 batchSize, 
	                                                 info,
	                                                 &internalDataInBytes,
	                                                 &workspaceInBytes));


	auto bufferOptions = torch::TensorOptions().dtype(torch::kByte).device(As_val.device());
	torch::Tensor buffer = torch::empty(workspaceInBytes, bufferOptions);

	auto xOptions = torch::TensorOptions().dtype(torch::kDouble).device(As_val.device());
	torch::Tensor x = torch::empty({(long)batchSize, (long)cols}, xOptions);
	
	// solve
	CUSOLVER_CHECK(cusolverSpDcsrqrsvBatched(cuSpHandle, rows, cols, nnz,
	                                         A_descr, pA_val, pA_rowPtr, pA_colInd,
	                                         b.data_ptr<double>(),
	                                         x.data_ptr<double>(),
	                                         batchSize,
	                                         info,
	                                         buffer.data_ptr<uint8_t>()));
 
	return x;
}

PYBIND11_MODULE(cusolver_qr_solver, m) {
	m.doc() = "Python bindings for cusolver QR solver";
    py::class_<CusolverQRSolver>(m, "CusolverQRSolver")
        .def(py::init<int64_t, const torch::Tensor&, const torch::Tensor&>())
        .def("factor_and_solve", &CusolverQRSolver::factorAndSolve);
};
