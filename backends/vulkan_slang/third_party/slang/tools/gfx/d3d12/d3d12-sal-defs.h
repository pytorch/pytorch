#pragma once

//
// The defines here are to allow compilers without support for SAL to not choke
// on any MS headers using these annotations
//

#if !defined(_In_)
#define _In_
#endif
#if !defined(_Out_)
#define _Out_
#endif
#if !defined(_Inout_)
#define _Inout_
#endif
#if !defined(_In_z_)
#define _In_z_
#endif
#if !defined(_Inout_z_)
#define _Inout_z_
#endif
#if !defined(_In_reads_)
#define _In_reads_(s)
#endif
#if !defined(_In_reads_z_)
#define _In_reads_z_(s)
#endif
#if !defined(_In_reads_or_z_)
#define _In_reads_or_z_(s)
#endif
#if !defined(_Out_writes_)
#define _Out_writes_(s)
#endif
#if !defined(_Out_writes_z_)
#define _Out_writes_z_(s)
#endif
#if !defined(_Inout_updates_)
#define _Inout_updates_(s)
#endif
#if !defined(_Inout_updates_z_)
#define _Inout_updates_z_(s)
#endif
#if !defined(_Out_writes_to_)
#define _Out_writes_to_(s, c)
#endif
#if !defined(_Inout_updates_to_)
#define _Inout_updates_to_(s, c)
#endif
#if !defined(_Inout_updates_all_)
#define _Inout_updates_all_(s)
#endif
#if !defined(_In_reads_to_ptr_)
#define _In_reads_to_ptr_(p)
#endif
#if !defined(_In_reads_to_ptr_z_)
#define _In_reads_to_ptr_z_(p)
#endif
#if !defined(_Out_writes_to_ptr_)
#define _Out_writes_to_ptr_(p)
#endif
#if !defined(_Out_writes_to_ptr_z_)
#define _Out_writes_to_ptr_z_(p)
#endif
#if !defined(_Outptr_)
#define _Outptr_
#endif
#if !defined(_Outptr_opt_)
#define _Outptr_opt_
#endif
#if !defined(_Outptr_result_maybenull_)
#define _Outptr_result_maybenull_
#endif
#if !defined(_Outptr_opt_result_maybenull_)
#define _Outptr_opt_result_maybenull_
#endif
#if !defined(_Outptr_result_z_)
#define _Outptr_result_z_
#endif
#if !defined(_COM_Outptr_)
#define _COM_Outptr_
#endif
#if !defined(_Outptr_result_buffer_)
#define _Outptr_result_buffer_(s)
#endif
#if !defined(_Outptr_result_buffer_to_)
#define _Outptr_result_buffer_to_(s, c)
#endif
#if !defined(_Result_nullonfailure_)
#define _Result_nullonfailure_
#endif
#if !defined(_Result_zeroonfailure_)
#define _Result_zeroonfailure_
#endif
#if !defined(_Outptr_result_nullonfailure_)
#define _Outptr_result_nullonfailure_
#endif
#if !defined(_Outptr_opt_result_nullonfailure_)
#define _Outptr_opt_result_nullonfailure_
#endif
#if !defined(_Outref_result_nullonfailure_)
#define _Outref_result_nullonfailure_
#endif
#if !defined(_Outref_)
#define _Outref_
#endif
#if !defined(_Outref_result_maybenull_)
#define _Outref_result_maybenull_
#endif
#if !defined(_Outref_result_buffer_)
#define _Outref_result_buffer_(s)
#endif
#if !defined(_Outref_result_bytebuffer_)
#define _Outref_result_bytebuffer_(s)
#endif
#if !defined(_Outref_result_buffer_to_)
#define _Outref_result_buffer_to_(s, c)
#endif
#if !defined(_Outref_result_bytebuffer_to_)
#define _Outref_result_bytebuffer_to_(s, c)
#endif
#if !defined(_Outref_result_buffer_all_)
#define _Outref_result_buffer_all_(s)
#endif
#if !defined(_Outref_result_bytebuffer_all_)
#define _Outref_result_bytebuffer_all_(s)
#endif
#if !defined(_Outref_result_buffer_maybenull_)
#define _Outref_result_buffer_maybenull_(s)
#endif
#if !defined(_Outref_result_bytebuffer_maybenull_)
#define _Outref_result_bytebuffer_maybenull_(s)
#endif
#if !defined(_Outref_result_buffer_to_maybenull_)
#define _Outref_result_buffer_to_maybenull_(s, c)
#endif
#if !defined(_Outref_result_bytebuffer_to_maybenull_)
#define _Outref_result_bytebuffer_to_maybenull_(s, c)
#endif
#if !defined(_Outref_result_buffer_all_maybenull_)
#define _Outref_result_buffer_all_maybenull_(s)
#endif
#if !defined(_Outref_result_bytebuffer_all_maybenull_)
#define _Outref_result_bytebuffer_all_maybenull_(s)
#endif
#if !defined(_Printf_format_string_)
#define _Printf_format_string_
#endif
#if !defined(_Scanf_format_string_)
#define _Scanf_format_string_
#endif
#if !defined(_Scanf_s_format_string_)
#define _Scanf_s_format_string_
#endif
#if !defined(_In_range_)
#define _In_range_(low, hi)
#endif
#if !defined(_Pre_equal_to_)
#define _Pre_equal_to_(expr)
#endif
#if !defined(_Struct_size_bytes_)
#define _Struct_size_bytes_(size)
#endif
#if !defined(_Called_from_function_class_)
#define _Called_from_function_class_(name)
#endif
#if !defined(_Check_return_)
#define _Check_return_
#endif
#if !defined(_Function_class_)
#define _Function_class_(name)
#endif
#if !defined(_Raises_SEH_exception_)
#define _Raises_SEH_exception_
#endif
#if !defined(_Maybe_raises_SEH_exception_)
#define _Maybe_raises_SEH_exception_
#endif
#if !defined(_Must_inspect_result_)
#define _Must_inspect_result_
#endif
#if !defined(_Use_decl_annotations_)
#define _Use_decl_annotations_
#endif
#if !defined(_Always_)
#define _Always_(anno_list)
#endif
#if !defined(_On_failure_)
#define _On_failure_(anno_list)
#endif
#if !defined(_Return_type_success_)
#define _Return_type_success_(expr)
#endif
#if !defined(_Success_)
#define _Success_(expr)
#endif
#if !defined(__analysis_assume)
#define __analysis_assume(expr)
#endif
