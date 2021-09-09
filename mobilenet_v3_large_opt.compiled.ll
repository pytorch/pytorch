; ModuleID = 'pytorch'
source_filename = "pytorch"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind
define i32 @wrapper(i8** readonly %0) local_unnamed_addr #0 {
wrapBB:
  %1 = alloca [0 x i64], align 8
  %2 = alloca [480 x float], align 4
  %3 = alloca [288 x float], align 4
  %4 = alloca [480 x float], align 4
  %5 = alloca [96 x float], align 4
  %6 = alloca [288 x float], align 4
  %7 = alloca [480 x float], align 4
  %8 = alloca [480 x float], align 4
  %9 = alloca [128 x float], align 4
  %10 = alloca [480 x float], align 4
  %11 = alloca [128 x float], align 4
  %12 = alloca [3 x i8*], align 8
  %13 = alloca [3 x i64], align 16
  %14 = alloca [8 x i64], align 8
  %15 = alloca [3 x i8], align 1
  %16 = bitcast i8** %0 to float**
  %17 = load float*, float** %16, align 8
  %18 = getelementptr i8*, i8** %0, i64 1
  %19 = bitcast i8** %18 to float**
  %20 = load float*, float** %19, align 8
  %21 = getelementptr i8*, i8** %0, i64 2
  %22 = bitcast i8** %21 to float**
  %23 = load float*, float** %22, align 8
  %24 = getelementptr i8*, i8** %0, i64 3
  %25 = bitcast i8** %24 to float**
  %26 = load float*, float** %25, align 8
  %27 = getelementptr i8*, i8** %0, i64 4
  %28 = bitcast i8** %27 to float**
  %29 = load float*, float** %28, align 8
  %30 = getelementptr i8*, i8** %0, i64 5
  %31 = bitcast i8** %30 to float**
  %32 = load float*, float** %31, align 8
  %33 = getelementptr i8*, i8** %0, i64 6
  %34 = bitcast i8** %33 to float**
  %35 = load float*, float** %34, align 8
  %36 = getelementptr i8*, i8** %0, i64 7
  %37 = bitcast i8** %36 to float**
  %38 = load float*, float** %37, align 8
  %39 = getelementptr i8*, i8** %0, i64 8
  %40 = bitcast i8** %39 to float**
  %41 = load float*, float** %40, align 8
  %42 = getelementptr i8*, i8** %0, i64 9
  %43 = bitcast i8** %42 to float**
  %44 = load float*, float** %43, align 8
  %45 = getelementptr i8*, i8** %0, i64 10
  %46 = bitcast i8** %45 to float**
  %47 = load float*, float** %46, align 8
  %48 = getelementptr i8*, i8** %0, i64 11
  %49 = bitcast i8** %48 to float**
  %50 = load float*, float** %49, align 8
  %51 = getelementptr i8*, i8** %0, i64 12
  %52 = bitcast i8** %51 to float**
  %53 = load float*, float** %52, align 8
  %54 = getelementptr i8*, i8** %0, i64 13
  %55 = bitcast i8** %54 to float**
  %56 = load float*, float** %55, align 8
  %57 = getelementptr i8*, i8** %0, i64 14
  %58 = bitcast i8** %57 to float**
  %59 = load float*, float** %58, align 8
  %60 = getelementptr i8*, i8** %0, i64 15
  %61 = bitcast i8** %60 to float**
  %62 = load float*, float** %61, align 8
  %63 = getelementptr i8*, i8** %0, i64 16
  %64 = bitcast i8** %63 to float**
  %65 = load float*, float** %64, align 8
  %66 = getelementptr i8*, i8** %0, i64 17
  %67 = bitcast i8** %66 to float**
  %68 = load float*, float** %67, align 8
  %69 = getelementptr i8*, i8** %0, i64 18
  %70 = bitcast i8** %69 to float**
  %71 = load float*, float** %70, align 8
  %72 = getelementptr i8*, i8** %0, i64 19
  %73 = bitcast i8** %72 to float**
  %74 = load float*, float** %73, align 8
  %75 = getelementptr i8*, i8** %0, i64 20
  %76 = bitcast i8** %75 to float**
  %77 = load float*, float** %76, align 8
  %78 = getelementptr i8*, i8** %0, i64 21
  %79 = bitcast i8** %78 to float**
  %80 = load float*, float** %79, align 8
  %81 = getelementptr i8*, i8** %0, i64 22
  %82 = bitcast i8** %81 to float**
  %83 = load float*, float** %82, align 8
  %84 = getelementptr i8*, i8** %0, i64 23
  %85 = bitcast i8** %84 to float**
  %86 = load float*, float** %85, align 8
  %87 = getelementptr i8*, i8** %0, i64 24
  %88 = bitcast i8** %87 to float**
  %89 = load float*, float** %88, align 8
  %90 = getelementptr i8*, i8** %0, i64 25
  %91 = bitcast i8** %90 to float**
  %92 = load float*, float** %91, align 8
  %93 = getelementptr i8*, i8** %0, i64 26
  %94 = bitcast i8** %93 to float**
  %95 = load float*, float** %94, align 8
  %96 = getelementptr i8*, i8** %0, i64 27
  %97 = bitcast i8** %96 to float**
  %98 = load float*, float** %97, align 8
  %99 = getelementptr i8*, i8** %0, i64 28
  %100 = bitcast i8** %99 to float**
  %101 = load float*, float** %100, align 8
  %102 = getelementptr i8*, i8** %0, i64 29
  %103 = bitcast i8** %102 to float**
  %104 = load float*, float** %103, align 8
  %105 = getelementptr i8*, i8** %0, i64 30
  %106 = bitcast i8** %105 to float**
  %107 = load float*, float** %106, align 8
  %108 = getelementptr i8*, i8** %0, i64 31
  %109 = bitcast i8** %108 to float**
  %110 = load float*, float** %109, align 8
  %111 = getelementptr i8*, i8** %0, i64 32
  %112 = bitcast i8** %111 to float**
  %113 = load float*, float** %112, align 8
  %114 = getelementptr i8*, i8** %0, i64 33
  %115 = bitcast i8** %114 to float**
  %116 = load float*, float** %115, align 8
  %117 = getelementptr i8*, i8** %0, i64 34
  %118 = bitcast i8** %117 to float**
  %119 = load float*, float** %118, align 8
  %120 = getelementptr i8*, i8** %0, i64 35
  %121 = bitcast i8** %120 to float**
  %122 = load float*, float** %121, align 8
  %123 = getelementptr i8*, i8** %0, i64 36
  %124 = bitcast i8** %123 to float**
  %125 = load float*, float** %124, align 8
  %126 = getelementptr i8*, i8** %0, i64 37
  %127 = bitcast i8** %126 to float**
  %128 = load float*, float** %127, align 8
  %129 = getelementptr i8*, i8** %0, i64 38
  %130 = bitcast i8** %129 to float**
  %131 = load float*, float** %130, align 8
  %132 = getelementptr i8*, i8** %0, i64 39
  %133 = bitcast i8** %132 to float**
  %134 = load float*, float** %133, align 8
  %135 = getelementptr i8*, i8** %0, i64 40
  %136 = bitcast i8** %135 to float**
  %137 = load float*, float** %136, align 8
  %138 = getelementptr i8*, i8** %0, i64 41
  %139 = bitcast i8** %138 to float**
  %140 = load float*, float** %139, align 8
  %141 = getelementptr i8*, i8** %0, i64 42
  %142 = bitcast i8** %141 to float**
  %143 = load float*, float** %142, align 8
  %144 = getelementptr i8*, i8** %0, i64 43
  %145 = bitcast i8** %144 to float**
  %146 = load float*, float** %145, align 8
  %147 = getelementptr i8*, i8** %0, i64 44
  %148 = bitcast i8** %147 to float**
  %149 = load float*, float** %148, align 8
  %150 = getelementptr i8*, i8** %0, i64 45
  %151 = bitcast i8** %150 to float**
  %152 = load float*, float** %151, align 8
  %153 = getelementptr i8*, i8** %0, i64 46
  %154 = bitcast i8** %153 to float**
  %155 = load float*, float** %154, align 8
  %156 = getelementptr i8*, i8** %0, i64 47
  %157 = bitcast i8** %156 to float**
  %158 = load float*, float** %157, align 8
  %159 = getelementptr i8*, i8** %0, i64 48
  %160 = bitcast i8** %159 to float**
  %161 = load float*, float** %160, align 8
  %162 = getelementptr i8*, i8** %0, i64 49
  %163 = bitcast i8** %162 to float**
  %164 = load float*, float** %163, align 8
  %165 = getelementptr i8*, i8** %0, i64 50
  %166 = bitcast i8** %165 to float**
  %167 = load float*, float** %166, align 8
  %168 = getelementptr i8*, i8** %0, i64 51
  %169 = bitcast i8** %168 to float**
  %170 = load float*, float** %169, align 8
  %171 = getelementptr i8*, i8** %0, i64 52
  %172 = bitcast i8** %171 to float**
  %173 = load float*, float** %172, align 8
  %174 = getelementptr i8*, i8** %0, i64 53
  %175 = bitcast i8** %174 to float**
  %176 = load float*, float** %175, align 8
  %177 = getelementptr i8*, i8** %0, i64 54
  %178 = bitcast i8** %177 to float**
  %179 = load float*, float** %178, align 8
  %180 = getelementptr i8*, i8** %0, i64 55
  %181 = bitcast i8** %180 to float**
  %182 = load float*, float** %181, align 8
  %183 = getelementptr i8*, i8** %0, i64 56
  %184 = bitcast i8** %183 to float**
  %185 = load float*, float** %184, align 8
  %186 = getelementptr i8*, i8** %0, i64 57
  %187 = bitcast i8** %186 to float**
  %188 = load float*, float** %187, align 8
  %189 = getelementptr i8*, i8** %0, i64 58
  %190 = bitcast i8** %189 to float**
  %191 = load float*, float** %190, align 8
  %192 = getelementptr i8*, i8** %0, i64 59
  %193 = bitcast i8** %192 to float**
  %194 = load float*, float** %193, align 8
  %195 = getelementptr i8*, i8** %0, i64 60
  %196 = bitcast i8** %195 to float**
  %197 = load float*, float** %196, align 8
  %198 = getelementptr i8*, i8** %0, i64 61
  %199 = bitcast i8** %198 to float**
  %200 = load float*, float** %199, align 8
  %201 = getelementptr i8*, i8** %0, i64 62
  %202 = bitcast i8** %201 to float**
  %203 = load float*, float** %202, align 8
  %204 = getelementptr i8*, i8** %0, i64 63
  %205 = bitcast i8** %204 to float**
  %206 = load float*, float** %205, align 8
  %207 = getelementptr i8*, i8** %0, i64 64
  %208 = bitcast i8** %207 to float**
  %209 = load float*, float** %208, align 8
  %210 = getelementptr i8*, i8** %0, i64 65
  %211 = bitcast i8** %210 to float**
  %212 = load float*, float** %211, align 8
  call void @llvm.experimental.noalias.scope.decl(metadata !0)
  call void @llvm.experimental.noalias.scope.decl(metadata !3)
  call void @llvm.experimental.noalias.scope.decl(metadata !5)
  call void @llvm.experimental.noalias.scope.decl(metadata !7)
  call void @llvm.experimental.noalias.scope.decl(metadata !9)
  call void @llvm.experimental.noalias.scope.decl(metadata !11)
  call void @llvm.experimental.noalias.scope.decl(metadata !13)
  call void @llvm.experimental.noalias.scope.decl(metadata !15)
  call void @llvm.experimental.noalias.scope.decl(metadata !17)
  call void @llvm.experimental.noalias.scope.decl(metadata !19)
  call void @llvm.experimental.noalias.scope.decl(metadata !21)
  call void @llvm.experimental.noalias.scope.decl(metadata !23)
  call void @llvm.experimental.noalias.scope.decl(metadata !25)
  call void @llvm.experimental.noalias.scope.decl(metadata !27)
  call void @llvm.experimental.noalias.scope.decl(metadata !29)
  call void @llvm.experimental.noalias.scope.decl(metadata !31)
  call void @llvm.experimental.noalias.scope.decl(metadata !33)
  call void @llvm.experimental.noalias.scope.decl(metadata !35)
  call void @llvm.experimental.noalias.scope.decl(metadata !37)
  call void @llvm.experimental.noalias.scope.decl(metadata !39)
  call void @llvm.experimental.noalias.scope.decl(metadata !41)
  call void @llvm.experimental.noalias.scope.decl(metadata !43)
  call void @llvm.experimental.noalias.scope.decl(metadata !45)
  call void @llvm.experimental.noalias.scope.decl(metadata !47)
  call void @llvm.experimental.noalias.scope.decl(metadata !49)
  call void @llvm.experimental.noalias.scope.decl(metadata !51)
  call void @llvm.experimental.noalias.scope.decl(metadata !53)
  call void @llvm.experimental.noalias.scope.decl(metadata !55)
  call void @llvm.experimental.noalias.scope.decl(metadata !57)
  call void @llvm.experimental.noalias.scope.decl(metadata !59)
  call void @llvm.experimental.noalias.scope.decl(metadata !61)
  call void @llvm.experimental.noalias.scope.decl(metadata !63)
  call void @llvm.experimental.noalias.scope.decl(metadata !65)
  call void @llvm.experimental.noalias.scope.decl(metadata !67)
  call void @llvm.experimental.noalias.scope.decl(metadata !69)
  call void @llvm.experimental.noalias.scope.decl(metadata !71)
  call void @llvm.experimental.noalias.scope.decl(metadata !73)
  call void @llvm.experimental.noalias.scope.decl(metadata !75)
  call void @llvm.experimental.noalias.scope.decl(metadata !77)
  call void @llvm.experimental.noalias.scope.decl(metadata !79)
  call void @llvm.experimental.noalias.scope.decl(metadata !81)
  call void @llvm.experimental.noalias.scope.decl(metadata !83)
  call void @llvm.experimental.noalias.scope.decl(metadata !85)
  call void @llvm.experimental.noalias.scope.decl(metadata !87)
  call void @llvm.experimental.noalias.scope.decl(metadata !89)
  call void @llvm.experimental.noalias.scope.decl(metadata !91)
  call void @llvm.experimental.noalias.scope.decl(metadata !93)
  call void @llvm.experimental.noalias.scope.decl(metadata !95)
  call void @llvm.experimental.noalias.scope.decl(metadata !97)
  call void @llvm.experimental.noalias.scope.decl(metadata !99)
  call void @llvm.experimental.noalias.scope.decl(metadata !101)
  call void @llvm.experimental.noalias.scope.decl(metadata !103)
  call void @llvm.experimental.noalias.scope.decl(metadata !105)
  call void @llvm.experimental.noalias.scope.decl(metadata !107)
  call void @llvm.experimental.noalias.scope.decl(metadata !109)
  call void @llvm.experimental.noalias.scope.decl(metadata !111)
  call void @llvm.experimental.noalias.scope.decl(metadata !113)
  call void @llvm.experimental.noalias.scope.decl(metadata !115)
  call void @llvm.experimental.noalias.scope.decl(metadata !117)
  call void @llvm.experimental.noalias.scope.decl(metadata !119)
  call void @llvm.experimental.noalias.scope.decl(metadata !121)
  call void @llvm.experimental.noalias.scope.decl(metadata !123)
  call void @llvm.experimental.noalias.scope.decl(metadata !125)
  call void @llvm.experimental.noalias.scope.decl(metadata !127)
  call void @llvm.experimental.noalias.scope.decl(metadata !129)
  call void @llvm.experimental.noalias.scope.decl(metadata !131)
  %savedstack = call i8* @llvm.stacksave()
  %213 = bitcast [0 x i64]* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 0, i8* %213)
  %214 = bitcast [480 x float]* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1920, i8* %214)
  %215 = bitcast [288 x float]* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1152, i8* %215)
  %216 = bitcast [480 x float]* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1920, i8* %216)
  %217 = bitcast [96 x float]* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 384, i8* %217)
  %218 = bitcast [288 x float]* %6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1152, i8* %218)
  %219 = bitcast [480 x float]* %7 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1920, i8* %219)
  %220 = bitcast [480 x float]* %8 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1920, i8* %220)
  %221 = bitcast [128 x float]* %9 to i8*
  call void @llvm.lifetime.start.p0i8(i64 512, i8* %221)
  %222 = bitcast [480 x float]* %10 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1920, i8* %222)
  %223 = bitcast [128 x float]* %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 512, i8* %223)
  %224 = bitcast [3 x i8*]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* %224)
  %225 = bitcast [3 x i64]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* %225)
  %226 = bitcast [8 x i64]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* %226)
  %227 = bitcast [3 x i8]* %15 to i8*
  call void @llvm.lifetime.start.p0i8(i64 3, i8* %227)
  %malloccall.i = tail call dereferenceable_or_null(188160) i8* @malloc(i64 188160) #0, !noalias !133
  %malloccall1.i = tail call dereferenceable_or_null(31360) i8* @malloc(i64 31360) #0, !noalias !133
  %malloccall2.i = tail call dereferenceable_or_null(188160) i8* @malloc(i64 188160) #0, !noalias !133
  %malloccall3.i = tail call dereferenceable_or_null(3840) i8* @malloc(i64 3840) #0, !noalias !133
  %malloccall4.i = tail call dereferenceable_or_null(188160) i8* @malloc(i64 188160) #0, !noalias !133
  %malloccall5.i = tail call dereferenceable_or_null(131712) i8* @malloc(i64 131712) #0, !noalias !133
  %malloccall6.i = tail call dereferenceable_or_null(188160) i8* @malloc(i64 188160) #0, !noalias !133
  %malloccall7.i = tail call dereferenceable_or_null(131712) i8* @malloc(i64 131712) #0, !noalias !133
  %malloccall8.i = tail call dereferenceable_or_null(526848) i8* @malloc(i64 526848) #0, !noalias !133
  %malloccall9.i = tail call dereferenceable_or_null(87808) i8* @malloc(i64 87808) #0, !noalias !133
  %malloccall10.i = tail call dereferenceable_or_null(2688) i8* @malloc(i64 2688) #0, !noalias !133
  %malloccall11.i = tail call dereferenceable_or_null(672) i8* @malloc(i64 672) #0, !noalias !133
  %malloccall12.i = tail call dereferenceable_or_null(526848) i8* @malloc(i64 526848) #0, !noalias !133
  %malloccall13.i = tail call dereferenceable_or_null(526848) i8* @malloc(i64 526848) #0, !noalias !133
  %malloccall14.i = tail call dereferenceable_or_null(376320) i8* @malloc(i64 376320) #0, !noalias !133
  %malloccall15.i = tail call dereferenceable_or_null(1920) i8* @malloc(i64 1920) #0, !noalias !133
  %malloccall16.i = tail call dereferenceable_or_null(156800) i8* @malloc(i64 156800) #0, !noalias !133
  %malloccall17.i = tail call dereferenceable_or_null(376320) i8* @malloc(i64 376320) #0, !noalias !133
  %malloccall18.i = tail call dereferenceable_or_null(376320) i8* @malloc(i64 376320) #0, !noalias !133
  %malloccall19.i = tail call dereferenceable_or_null(62720) i8* @malloc(i64 62720) #0, !noalias !133
  %malloccall20.i = tail call dereferenceable_or_null(376320) i8* @malloc(i64 376320) #0, !noalias !133
  %malloccall21.i = tail call dereferenceable_or_null(1920) i8* @malloc(i64 1920) #0, !noalias !133
  %malloccall22.i = tail call dereferenceable_or_null(144256) i8* @malloc(i64 144256) #0, !noalias !133
  %malloccall23.i = tail call dereferenceable_or_null(144256) i8* @malloc(i64 144256) #0, !noalias !133
  %malloccall24.i = tail call dereferenceable_or_null(156800) i8* @malloc(i64 156800) #0, !noalias !133
  %malloccall25.i = tail call dereferenceable_or_null(526848) i8* @malloc(i64 526848) #0, !noalias !133
  %malloccall26.i = tail call dereferenceable_or_null(144256) i8* @malloc(i64 144256) #0, !noalias !133
  %malloccall27.i = tail call dereferenceable_or_null(62720) i8* @malloc(i64 62720) #0, !noalias !133
  %malloccall28.i = tail call dereferenceable_or_null(144256) i8* @malloc(i64 144256) #0, !noalias !133
  %malloccall29.i = tail call dereferenceable_or_null(376320) i8* @malloc(i64 376320) #0, !noalias !133
  %malloccall30.i = tail call dereferenceable_or_null(144256) i8* @malloc(i64 144256) #0, !noalias !133
  %malloccall31.i = tail call dereferenceable_or_null(144256) i8* @malloc(i64 144256) #0, !noalias !133
  %malloccall32.i = tail call dereferenceable_or_null(125440) i8* @malloc(i64 125440) #0, !noalias !133
  %malloccall33.i = tail call dereferenceable_or_null(526848) i8* @malloc(i64 526848) #0, !noalias !133
  %malloccall34.i = tail call dereferenceable_or_null(2688) i8* @malloc(i64 2688) #0, !noalias !133
  %malloccall35.i = tail call dereferenceable_or_null(62720) i8* @malloc(i64 62720) #0, !noalias !133
  %malloccall36.i = tail call dereferenceable_or_null(156800) i8* @malloc(i64 156800) #0, !noalias !133
  %malloccall37.i = tail call dereferenceable_or_null(903168) i8* @malloc(i64 903168) #0, !noalias !133
  %malloccall38.i = tail call dereferenceable_or_null(526848) i8* @malloc(i64 526848) #0, !noalias !133
  %malloccall39.i = tail call dereferenceable_or_null(156800) i8* @malloc(i64 156800) #0, !noalias !133
  %malloccall40.i = tail call dereferenceable_or_null(960) i8* @malloc(i64 960) #0, !noalias !133
  %malloccall41.i = tail call dereferenceable_or_null(802816) i8* @malloc(i64 802816) #0, !noalias !133
  %228 = bitcast i8* %malloccall41.i to float*
  %malloccall42.i = tail call dereferenceable_or_null(188160) i8* @malloc(i64 188160) #0, !noalias !133
  %malloccall43.i = tail call dereferenceable_or_null(188160) i8* @malloc(i64 188160) #0, !noalias !133
  %malloccall44.i = tail call dereferenceable_or_null(125440) i8* @malloc(i64 125440) #0, !noalias !133
  %malloccall45.i = tail call dereferenceable_or_null(144256) i8* @malloc(i64 144256) #0, !noalias !133
  %malloccall46.i = tail call dereferenceable_or_null(125440) i8* @malloc(i64 125440) #0, !noalias !133
  %malloccall47.i = tail call dereferenceable_or_null(62720) i8* @malloc(i64 62720) #0, !noalias !133
  %malloccall48.i = tail call dereferenceable_or_null(802816) i8* @malloc(i64 802816) #0, !noalias !133
  %malloccall49.i = tail call dereferenceable_or_null(144256) i8* @malloc(i64 144256) #0, !noalias !133
  %malloccall50.i = tail call dereferenceable_or_null(188160) i8* @malloc(i64 188160) #0, !noalias !133
  %malloccall51.i = tail call dereferenceable_or_null(3840) i8* @malloc(i64 3840) #0, !noalias !133
  %malloccall52.i = tail call dereferenceable_or_null(752640) i8* @malloc(i64 752640) #0, !noalias !133
  %malloccall53.i = tail call dereferenceable_or_null(131712) i8* @malloc(i64 131712) #0, !noalias !133
  %malloccall54.i = tail call dereferenceable_or_null(3211264) i8* @malloc(i64 3211264) #0, !noalias !133
  %malloccall55.i = tail call dereferenceable_or_null(3840) i8* @malloc(i64 3840) #0, !noalias !133
  %malloccall56.i = tail call dereferenceable_or_null(87808) i8* @malloc(i64 87808) #0, !noalias !133
  %malloccall57.i = tail call dereferenceable_or_null(125440) i8* @malloc(i64 125440) #0, !noalias !133
  %malloccall58.i = tail call dereferenceable_or_null(62720) i8* @malloc(i64 62720) #0, !noalias !133
  %malloccall59.i = tail call dereferenceable_or_null(2688) i8* @malloc(i64 2688) #0, !noalias !133
  %malloccall60.i = tail call dereferenceable_or_null(62720) i8* @malloc(i64 62720) #0, !noalias !133
  %malloccall61.i = tail call dereferenceable_or_null(526848) i8* @malloc(i64 526848) #0, !noalias !133
  %malloccall62.i = tail call dereferenceable_or_null(376320) i8* @malloc(i64 376320) #0, !noalias !133
  %malloccall63.i = tail call dereferenceable_or_null(376320) i8* @malloc(i64 376320) #0, !noalias !133
  %malloccall64.i = tail call dereferenceable_or_null(802816) i8* @malloc(i64 802816) #0, !noalias !133
  %malloccall65.i = tail call dereferenceable_or_null(802816) i8* @malloc(i64 802816) #0, !noalias !133
  %malloccall66.i = tail call dereferenceable_or_null(802816) i8* @malloc(i64 802816) #0, !noalias !133
  %229 = bitcast i8* %malloccall66.i to float*
  %malloccall67.i = tail call dereferenceable_or_null(672) i8* @malloc(i64 672) #0, !noalias !133
  %malloccall68.i = tail call dereferenceable_or_null(188160) i8* @malloc(i64 188160) #0, !noalias !133
  %malloccall69.i = tail call dereferenceable_or_null(802816) i8* @malloc(i64 802816) #0, !noalias !133
  %malloccall70.i = tail call dereferenceable_or_null(903168) i8* @malloc(i64 903168) #0, !noalias !133
  %malloccall71.i = tail call dereferenceable_or_null(376320) i8* @malloc(i64 376320) #0, !noalias !133
  %malloccall72.i = tail call dereferenceable_or_null(301056) i8* @malloc(i64 301056) #0, !noalias !133
  %malloccall73.i = tail call dereferenceable_or_null(752640) i8* @malloc(i64 752640) #0, !noalias !133
  %malloccall74.i = tail call dereferenceable_or_null(188160) i8* @malloc(i64 188160) #0, !noalias !133
  %malloccall75.i = tail call dereferenceable_or_null(225792) i8* @malloc(i64 225792) #0, !noalias !133
  %malloccall76.i = tail call dereferenceable_or_null(301056) i8* @malloc(i64 301056) #0, !noalias !133
  %malloccall77.i = tail call dereferenceable_or_null(225792) i8* @malloc(i64 225792) #0, !noalias !133
  %malloccall78.i = tail call dereferenceable_or_null(125440) i8* @malloc(i64 125440) #0, !noalias !133
  %malloccall79.i = tail call dereferenceable_or_null(376320) i8* @malloc(i64 376320) #0, !noalias !133
  %malloccall80.i = tail call dereferenceable_or_null(301056) i8* @malloc(i64 301056) #0, !noalias !133
  %malloccall81.i = tail call dereferenceable_or_null(903168) i8* @malloc(i64 903168) #0, !noalias !133
  %malloccall82.i = tail call dereferenceable_or_null(87808) i8* @malloc(i64 87808) #0, !noalias !133
  %malloccall83.i = tail call dereferenceable_or_null(31360) i8* @malloc(i64 31360) #0, !noalias !133
  %malloccall84.i = tail call dereferenceable_or_null(188160) i8* @malloc(i64 188160) #0, !noalias !133
  %malloccall85.i = tail call dereferenceable_or_null(31360) i8* @malloc(i64 31360) #0, !noalias !133
  %malloccall86.i = tail call dereferenceable_or_null(376320) i8* @malloc(i64 376320) #0, !noalias !133
  %malloccall87.i = tail call dereferenceable_or_null(376320) i8* @malloc(i64 376320) #0, !noalias !133
  %malloccall88.i = tail call dereferenceable_or_null(2688) i8* @malloc(i64 2688) #0, !noalias !133
  %malloccall89.i = tail call dereferenceable_or_null(188160) i8* @malloc(i64 188160) #0, !noalias !133
  %malloccall90.i = tail call dereferenceable_or_null(62720) i8* @malloc(i64 62720) #0, !noalias !133
  %malloccall91.i = tail call dereferenceable_or_null(188160) i8* @malloc(i64 188160) #0, !noalias !133
  %malloccall92.i = tail call dereferenceable_or_null(960) i8* @malloc(i64 960) #0, !noalias !133
  %malloccall93.i = tail call dereferenceable_or_null(3840) i8* @malloc(i64 3840) #0, !noalias !133
  %malloccall94.i = tail call dereferenceable_or_null(31360) i8* @malloc(i64 31360) #0, !noalias !133
  %malloccall95.i = tail call dereferenceable_or_null(31360) i8* @malloc(i64 31360) #0, !noalias !133
  %malloccall96.i = tail call dereferenceable_or_null(188160) i8* @malloc(i64 188160) #0, !noalias !133
  %malloccall97.i = tail call dereferenceable_or_null(188160) i8* @malloc(i64 188160) #0, !noalias !133
  %malloccall98.i = tail call dereferenceable_or_null(3840) i8* @malloc(i64 3840) #0, !noalias !133
  %malloccall99.i = tail call dereferenceable_or_null(3840) i8* @malloc(i64 3840) #0, !noalias !133
  %malloccall100.i = tail call dereferenceable_or_null(5120) i8* @malloc(i64 5120) #0, !noalias !133
  %malloccall101.i = tail call dereferenceable_or_null(5120) i8* @malloc(i64 5120) #0, !noalias !133
  %.sub14.i = getelementptr inbounds [0 x i64], [0 x i64]* %1, i64 0, i64 0
  %.sub13.i = getelementptr inbounds [3 x i8], [3 x i8]* %15, i64 0, i64 0
  %.sub12.i = getelementptr inbounds [8 x i64], [8 x i64]* %14, i64 0, i64 0
  %.sub11.i = getelementptr inbounds [3 x i64], [3 x i64]* %13, i64 0, i64 0
  %.sub10.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %12, i64 0, i64 0
  store i8* %malloccall41.i, i8** %.sub10.i, align 8, !noalias !133
  store i8 6, i8* %.sub13.i, align 1, !noalias !133
  %230 = bitcast [8 x i64]* %14 to <4 x i64>*
  store <4 x i64> <i64 1, i64 16, i64 112, i64 112>, <4 x i64>* %230, align 8, !noalias !133
  %231 = getelementptr inbounds [3 x i8*], [3 x i8*]* %12, i64 0, i64 1
  %232 = bitcast i8** %231 to float**
  store float* %17, float** %232, align 8, !noalias !133
  %233 = getelementptr inbounds [3 x i8], [3 x i8]* %15, i64 0, i64 1
  store i8 6, i8* %233, align 1, !noalias !133
  %234 = bitcast [3 x i64]* %13 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %234, align 16, !noalias !133
  %235 = getelementptr inbounds [8 x i64], [8 x i64]* %14, i64 0, i64 4
  %236 = bitcast i64* %235 to <4 x i64>*
  store <4 x i64> <i64 1, i64 3, i64 224, i64 224>, <4 x i64>* %236, align 8, !noalias !133
  %237 = getelementptr inbounds [3 x i8*], [3 x i8*]* %12, i64 0, i64 2
  %238 = bitcast i8** %237 to float**
  store float* %23, float** %238, align 8, !noalias !133
  %239 = getelementptr inbounds [3 x i8], [3 x i8]* %15, i64 0, i64 2
  store i8 6, i8* %239, align 1, !noalias !133
  %240 = getelementptr inbounds [3 x i64], [3 x i64]* %13, i64 0, i64 2
  store i64 0, i64* %240, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub10.i, i64* nonnull %.sub11.i, i64* nonnull %.sub12.i, i8* nonnull %.sub13.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !134
  br label %cond102.preheader.i

cond102.preheader.i:                              ; preds = %exit104.i, %wrapBB
  %241 = phi i64 [ 0, %wrapBB ], [ %536, %exit104.i ]
  %242 = mul nuw nsw i64 %241, 12544
  br label %cond105.preheader.i

exit.i:                                           ; preds = %exit104.i
  %243 = bitcast i8* %malloccall.i to float*
  %244 = bitcast i8* %malloccall1.i to float*
  %245 = bitcast i8* %malloccall2.i to float*
  %246 = bitcast i8* %malloccall4.i to float*
  %247 = bitcast i8* %malloccall5.i to float*
  %248 = bitcast i8* %malloccall6.i to float*
  %249 = bitcast i8* %malloccall7.i to float*
  %250 = bitcast i8* %malloccall8.i to float*
  %251 = bitcast i8* %malloccall9.i to float*
  %252 = bitcast i8* %malloccall10.i to float*
  %253 = bitcast i8* %malloccall12.i to float*
  %254 = bitcast i8* %malloccall13.i to float*
  %255 = bitcast i8* %malloccall14.i to float*
  %256 = bitcast i8* %malloccall15.i to float*
  %257 = bitcast i8* %malloccall16.i to float*
  %258 = bitcast i8* %malloccall17.i to float*
  %259 = bitcast i8* %malloccall18.i to float*
  %260 = bitcast i8* %malloccall19.i to float*
  %261 = bitcast i8* %malloccall20.i to float*
  %262 = bitcast i8* %malloccall22.i to float*
  %263 = bitcast i8* %malloccall23.i to float*
  %264 = bitcast i8* %malloccall24.i to float*
  %265 = bitcast i8* %malloccall25.i to float*
  %266 = bitcast i8* %malloccall26.i to float*
  %267 = bitcast i8* %malloccall27.i to float*
  %268 = bitcast i8* %malloccall28.i to float*
  %269 = bitcast i8* %malloccall29.i to float*
  %270 = bitcast i8* %malloccall30.i to float*
  %271 = bitcast i8* %malloccall31.i to float*
  %272 = bitcast i8* %malloccall32.i to float*
  %273 = bitcast i8* %malloccall33.i to float*
  %274 = bitcast i8* %malloccall35.i to float*
  %275 = bitcast i8* %malloccall36.i to float*
  %276 = bitcast i8* %malloccall38.i to float*
  %277 = bitcast i8* %malloccall39.i to float*
  %278 = bitcast i8* %malloccall42.i to float*
  %279 = bitcast i8* %malloccall43.i to float*
  %280 = bitcast i8* %malloccall44.i to float*
  %281 = bitcast i8* %malloccall45.i to float*
  %282 = bitcast i8* %malloccall46.i to float*
  %283 = bitcast i8* %malloccall47.i to float*
  %284 = bitcast i8* %malloccall49.i to float*
  %285 = bitcast i8* %malloccall50.i to float*
  %286 = bitcast i8* %malloccall52.i to float*
  %287 = bitcast i8* %malloccall53.i to float*
  %288 = bitcast i8* %malloccall55.i to float*
  %289 = bitcast i8* %malloccall56.i to float*
  %290 = bitcast i8* %malloccall57.i to float*
  %291 = bitcast i8* %malloccall58.i to float*
  %292 = bitcast i8* %malloccall60.i to float*
  %293 = bitcast i8* %malloccall61.i to float*
  %294 = bitcast i8* %malloccall62.i to float*
  %295 = bitcast i8* %malloccall64.i to float*
  %296 = bitcast i8* %malloccall65.i to float*
  %297 = bitcast i8* %malloccall68.i to float*
  %298 = bitcast i8* %malloccall72.i to float*
  %299 = bitcast i8* %malloccall73.i to float*
  %300 = bitcast i8* %malloccall74.i to float*
  %301 = bitcast i8* %malloccall75.i to float*
  %302 = bitcast i8* %malloccall76.i to float*
  %303 = bitcast i8* %malloccall77.i to float*
  %304 = bitcast i8* %malloccall78.i to float*
  %305 = bitcast i8* %malloccall79.i to float*
  %306 = bitcast i8* %malloccall80.i to float*
  %307 = bitcast i8* %malloccall82.i to float*
  %308 = bitcast i8* %malloccall83.i to float*
  %309 = bitcast i8* %malloccall84.i to float*
  %310 = bitcast i8* %malloccall85.i to float*
  %311 = bitcast i8* %malloccall86.i to float*
  %312 = bitcast i8* %malloccall87.i to float*
  %313 = bitcast i8* %malloccall88.i to float*
  %314 = bitcast i8* %malloccall89.i to float*
  %315 = bitcast i8* %malloccall90.i to float*
  %316 = bitcast i8* %malloccall91.i to float*
  %317 = bitcast i8* %malloccall93.i to float*
  %318 = bitcast i8* %malloccall94.i to float*
  %319 = bitcast i8* %malloccall95.i to float*
  %320 = bitcast i8* %malloccall96.i to float*
  %321 = bitcast i8* %malloccall97.i to float*
  %322 = bitcast i8* %malloccall100.i to float*
  %323 = alloca [3 x i8*], align 8
  %324 = alloca [3 x i64], align 16
  %325 = alloca [8 x i64], align 8
  %326 = alloca [3 x i8], align 1
  %.sub18.i = getelementptr inbounds [3 x i8], [3 x i8]* %326, i64 0, i64 0
  %.sub17.i = getelementptr inbounds [8 x i64], [8 x i64]* %325, i64 0, i64 0
  %.sub16.i = getelementptr inbounds [3 x i64], [3 x i64]* %324, i64 0, i64 0
  %.sub15.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %323, i64 0, i64 0
  store i8* %malloccall69.i, i8** %.sub15.i, align 8, !noalias !133
  store i8 6, i8* %.sub18.i, align 1, !noalias !133
  %327 = bitcast [8 x i64]* %325 to <4 x i64>*
  store <4 x i64> <i64 1, i64 16, i64 112, i64 112>, <4 x i64>* %327, align 8, !noalias !133
  %328 = getelementptr inbounds [3 x i8*], [3 x i8*]* %323, i64 0, i64 1
  store i8* %malloccall66.i, i8** %328, align 8, !noalias !133
  %329 = getelementptr inbounds [3 x i8], [3 x i8]* %326, i64 0, i64 1
  store i8 6, i8* %329, align 1, !noalias !133
  %330 = bitcast [3 x i64]* %324 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %330, align 16, !noalias !133
  %331 = getelementptr inbounds [8 x i64], [8 x i64]* %325, i64 0, i64 4
  %332 = bitcast i64* %331 to <4 x i64>*
  store <4 x i64> <i64 1, i64 16, i64 112, i64 112>, <4 x i64>* %332, align 8, !noalias !133
  %333 = getelementptr inbounds [3 x i8*], [3 x i8*]* %323, i64 0, i64 2
  %334 = bitcast i8** %333 to float**
  store float* %26, float** %334, align 8, !noalias !133
  %335 = getelementptr inbounds [3 x i8], [3 x i8]* %326, i64 0, i64 2
  store i8 6, i8* %335, align 1, !noalias !133
  %336 = getelementptr inbounds [3 x i64], [3 x i64]* %324, i64 0, i64 2
  store i64 0, i64* %336, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub15.i, i64* nonnull %.sub16.i, i64* nonnull %.sub17.i, i8* nonnull %.sub18.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %337 = alloca [3 x i8*], align 8
  %338 = alloca [3 x i64], align 16
  %339 = alloca [8 x i64], align 8
  %340 = alloca [3 x i8], align 1
  %.sub23.i = getelementptr inbounds [3 x i8], [3 x i8]* %340, i64 0, i64 0
  %.sub22.i = getelementptr inbounds [8 x i64], [8 x i64]* %339, i64 0, i64 0
  %.sub21.i = getelementptr inbounds [3 x i64], [3 x i64]* %338, i64 0, i64 0
  %.sub20.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %337, i64 0, i64 0
  store i8* %malloccall64.i, i8** %.sub20.i, align 8, !noalias !133
  store i8 6, i8* %.sub23.i, align 1, !noalias !133
  %341 = bitcast [8 x i64]* %339 to <4 x i64>*
  store <4 x i64> <i64 1, i64 16, i64 112, i64 112>, <4 x i64>* %341, align 8, !noalias !133
  %342 = getelementptr inbounds [3 x i8*], [3 x i8*]* %337, i64 0, i64 1
  store i8* %malloccall69.i, i8** %342, align 8, !noalias !133
  %343 = getelementptr inbounds [3 x i8], [3 x i8]* %340, i64 0, i64 1
  store i8 6, i8* %343, align 1, !noalias !133
  %344 = bitcast [3 x i64]* %338 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %344, align 16, !noalias !133
  %345 = getelementptr inbounds [8 x i64], [8 x i64]* %339, i64 0, i64 4
  %346 = bitcast i64* %345 to <4 x i64>*
  store <4 x i64> <i64 1, i64 16, i64 112, i64 112>, <4 x i64>* %346, align 8, !noalias !133
  %347 = getelementptr inbounds [3 x i8*], [3 x i8*]* %337, i64 0, i64 2
  %348 = bitcast i8** %347 to float**
  store float* %29, float** %348, align 8, !noalias !133
  %349 = getelementptr inbounds [3 x i8], [3 x i8]* %340, i64 0, i64 2
  store i8 6, i8* %349, align 1, !noalias !133
  %350 = getelementptr inbounds [3 x i64], [3 x i64]* %338, i64 0, i64 2
  store i64 0, i64* %350, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub20.i, i64* nonnull %.sub21.i, i64* nonnull %.sub22.i, i8* nonnull %.sub23.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond114.preheader.i

cond105.preheader.i:                              ; preds = %cond105.preheader.i, %cond102.preheader.i
  %351 = phi i64 [ 0, %cond102.preheader.i ], [ %535, %cond105.preheader.i ]
  %352 = mul nuw nsw i64 %351, 112
  %353 = add nuw nsw i64 %352, %242
  %354 = getelementptr float, float* %228, i64 %353
  %355 = getelementptr float, float* %229, i64 %353
  %356 = bitcast float* %354 to <8 x float>*
  %357 = load <8 x float>, <8 x float>* %356, align 4, !noalias !133
  %358 = fadd <8 x float> %357, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %359 = fcmp olt <8 x float> %358, zeroinitializer
  %360 = select <8 x i1> %359, <8 x float> zeroinitializer, <8 x float> %358
  %361 = fcmp ogt <8 x float> %360, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %362 = select <8 x i1> %361, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %360
  %363 = fmul <8 x float> %357, %362
  %364 = fdiv <8 x float> %363, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %365 = bitcast float* %355 to <8 x float>*
  store <8 x float> %364, <8 x float>* %365, align 4, !noalias !133
  %366 = or i64 %353, 8
  %367 = getelementptr float, float* %228, i64 %366
  %368 = getelementptr float, float* %229, i64 %366
  %369 = bitcast float* %367 to <8 x float>*
  %370 = load <8 x float>, <8 x float>* %369, align 4, !noalias !133
  %371 = fadd <8 x float> %370, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %372 = fcmp olt <8 x float> %371, zeroinitializer
  %373 = select <8 x i1> %372, <8 x float> zeroinitializer, <8 x float> %371
  %374 = fcmp ogt <8 x float> %373, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %375 = select <8 x i1> %374, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %373
  %376 = fmul <8 x float> %370, %375
  %377 = fdiv <8 x float> %376, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %378 = bitcast float* %368 to <8 x float>*
  store <8 x float> %377, <8 x float>* %378, align 4, !noalias !133
  %379 = add nuw nsw i64 %353, 16
  %380 = getelementptr float, float* %228, i64 %379
  %381 = getelementptr float, float* %229, i64 %379
  %382 = bitcast float* %380 to <8 x float>*
  %383 = load <8 x float>, <8 x float>* %382, align 4, !noalias !133
  %384 = fadd <8 x float> %383, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %385 = fcmp olt <8 x float> %384, zeroinitializer
  %386 = select <8 x i1> %385, <8 x float> zeroinitializer, <8 x float> %384
  %387 = fcmp ogt <8 x float> %386, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %388 = select <8 x i1> %387, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %386
  %389 = fmul <8 x float> %383, %388
  %390 = fdiv <8 x float> %389, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %391 = bitcast float* %381 to <8 x float>*
  store <8 x float> %390, <8 x float>* %391, align 4, !noalias !133
  %392 = add nuw nsw i64 %353, 24
  %393 = getelementptr float, float* %228, i64 %392
  %394 = getelementptr float, float* %229, i64 %392
  %395 = bitcast float* %393 to <8 x float>*
  %396 = load <8 x float>, <8 x float>* %395, align 4, !noalias !133
  %397 = fadd <8 x float> %396, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %398 = fcmp olt <8 x float> %397, zeroinitializer
  %399 = select <8 x i1> %398, <8 x float> zeroinitializer, <8 x float> %397
  %400 = fcmp ogt <8 x float> %399, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %401 = select <8 x i1> %400, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %399
  %402 = fmul <8 x float> %396, %401
  %403 = fdiv <8 x float> %402, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %404 = bitcast float* %394 to <8 x float>*
  store <8 x float> %403, <8 x float>* %404, align 4, !noalias !133
  %405 = add nuw nsw i64 %353, 32
  %406 = getelementptr float, float* %228, i64 %405
  %407 = getelementptr float, float* %229, i64 %405
  %408 = bitcast float* %406 to <8 x float>*
  %409 = load <8 x float>, <8 x float>* %408, align 4, !noalias !133
  %410 = fadd <8 x float> %409, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %411 = fcmp olt <8 x float> %410, zeroinitializer
  %412 = select <8 x i1> %411, <8 x float> zeroinitializer, <8 x float> %410
  %413 = fcmp ogt <8 x float> %412, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %414 = select <8 x i1> %413, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %412
  %415 = fmul <8 x float> %409, %414
  %416 = fdiv <8 x float> %415, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %417 = bitcast float* %407 to <8 x float>*
  store <8 x float> %416, <8 x float>* %417, align 4, !noalias !133
  %418 = add nuw nsw i64 %353, 40
  %419 = getelementptr float, float* %228, i64 %418
  %420 = getelementptr float, float* %229, i64 %418
  %421 = bitcast float* %419 to <8 x float>*
  %422 = load <8 x float>, <8 x float>* %421, align 4, !noalias !133
  %423 = fadd <8 x float> %422, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %424 = fcmp olt <8 x float> %423, zeroinitializer
  %425 = select <8 x i1> %424, <8 x float> zeroinitializer, <8 x float> %423
  %426 = fcmp ogt <8 x float> %425, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %427 = select <8 x i1> %426, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %425
  %428 = fmul <8 x float> %422, %427
  %429 = fdiv <8 x float> %428, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %430 = bitcast float* %420 to <8 x float>*
  store <8 x float> %429, <8 x float>* %430, align 4, !noalias !133
  %431 = add nuw nsw i64 %353, 48
  %432 = getelementptr float, float* %228, i64 %431
  %433 = getelementptr float, float* %229, i64 %431
  %434 = bitcast float* %432 to <8 x float>*
  %435 = load <8 x float>, <8 x float>* %434, align 4, !noalias !133
  %436 = fadd <8 x float> %435, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %437 = fcmp olt <8 x float> %436, zeroinitializer
  %438 = select <8 x i1> %437, <8 x float> zeroinitializer, <8 x float> %436
  %439 = fcmp ogt <8 x float> %438, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %440 = select <8 x i1> %439, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %438
  %441 = fmul <8 x float> %435, %440
  %442 = fdiv <8 x float> %441, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %443 = bitcast float* %433 to <8 x float>*
  store <8 x float> %442, <8 x float>* %443, align 4, !noalias !133
  %444 = add nuw nsw i64 %353, 56
  %445 = getelementptr float, float* %228, i64 %444
  %446 = getelementptr float, float* %229, i64 %444
  %447 = bitcast float* %445 to <8 x float>*
  %448 = load <8 x float>, <8 x float>* %447, align 4, !noalias !133
  %449 = fadd <8 x float> %448, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %450 = fcmp olt <8 x float> %449, zeroinitializer
  %451 = select <8 x i1> %450, <8 x float> zeroinitializer, <8 x float> %449
  %452 = fcmp ogt <8 x float> %451, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %453 = select <8 x i1> %452, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %451
  %454 = fmul <8 x float> %448, %453
  %455 = fdiv <8 x float> %454, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %456 = bitcast float* %446 to <8 x float>*
  store <8 x float> %455, <8 x float>* %456, align 4, !noalias !133
  %457 = add nuw nsw i64 %353, 64
  %458 = getelementptr float, float* %228, i64 %457
  %459 = getelementptr float, float* %229, i64 %457
  %460 = bitcast float* %458 to <8 x float>*
  %461 = load <8 x float>, <8 x float>* %460, align 4, !noalias !133
  %462 = fadd <8 x float> %461, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %463 = fcmp olt <8 x float> %462, zeroinitializer
  %464 = select <8 x i1> %463, <8 x float> zeroinitializer, <8 x float> %462
  %465 = fcmp ogt <8 x float> %464, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %466 = select <8 x i1> %465, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %464
  %467 = fmul <8 x float> %461, %466
  %468 = fdiv <8 x float> %467, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %469 = bitcast float* %459 to <8 x float>*
  store <8 x float> %468, <8 x float>* %469, align 4, !noalias !133
  %470 = add nuw nsw i64 %353, 72
  %471 = getelementptr float, float* %228, i64 %470
  %472 = getelementptr float, float* %229, i64 %470
  %473 = bitcast float* %471 to <8 x float>*
  %474 = load <8 x float>, <8 x float>* %473, align 4, !noalias !133
  %475 = fadd <8 x float> %474, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %476 = fcmp olt <8 x float> %475, zeroinitializer
  %477 = select <8 x i1> %476, <8 x float> zeroinitializer, <8 x float> %475
  %478 = fcmp ogt <8 x float> %477, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %479 = select <8 x i1> %478, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %477
  %480 = fmul <8 x float> %474, %479
  %481 = fdiv <8 x float> %480, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %482 = bitcast float* %472 to <8 x float>*
  store <8 x float> %481, <8 x float>* %482, align 4, !noalias !133
  %483 = add nuw nsw i64 %353, 80
  %484 = getelementptr float, float* %228, i64 %483
  %485 = getelementptr float, float* %229, i64 %483
  %486 = bitcast float* %484 to <8 x float>*
  %487 = load <8 x float>, <8 x float>* %486, align 4, !noalias !133
  %488 = fadd <8 x float> %487, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %489 = fcmp olt <8 x float> %488, zeroinitializer
  %490 = select <8 x i1> %489, <8 x float> zeroinitializer, <8 x float> %488
  %491 = fcmp ogt <8 x float> %490, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %492 = select <8 x i1> %491, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %490
  %493 = fmul <8 x float> %487, %492
  %494 = fdiv <8 x float> %493, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %495 = bitcast float* %485 to <8 x float>*
  store <8 x float> %494, <8 x float>* %495, align 4, !noalias !133
  %496 = add nuw nsw i64 %353, 88
  %497 = getelementptr float, float* %228, i64 %496
  %498 = getelementptr float, float* %229, i64 %496
  %499 = bitcast float* %497 to <8 x float>*
  %500 = load <8 x float>, <8 x float>* %499, align 4, !noalias !133
  %501 = fadd <8 x float> %500, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %502 = fcmp olt <8 x float> %501, zeroinitializer
  %503 = select <8 x i1> %502, <8 x float> zeroinitializer, <8 x float> %501
  %504 = fcmp ogt <8 x float> %503, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %505 = select <8 x i1> %504, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %503
  %506 = fmul <8 x float> %500, %505
  %507 = fdiv <8 x float> %506, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %508 = bitcast float* %498 to <8 x float>*
  store <8 x float> %507, <8 x float>* %508, align 4, !noalias !133
  %509 = add nuw nsw i64 %353, 96
  %510 = getelementptr float, float* %228, i64 %509
  %511 = getelementptr float, float* %229, i64 %509
  %512 = bitcast float* %510 to <8 x float>*
  %513 = load <8 x float>, <8 x float>* %512, align 4, !noalias !133
  %514 = fadd <8 x float> %513, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %515 = fcmp olt <8 x float> %514, zeroinitializer
  %516 = select <8 x i1> %515, <8 x float> zeroinitializer, <8 x float> %514
  %517 = fcmp ogt <8 x float> %516, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %518 = select <8 x i1> %517, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %516
  %519 = fmul <8 x float> %513, %518
  %520 = fdiv <8 x float> %519, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %521 = bitcast float* %511 to <8 x float>*
  store <8 x float> %520, <8 x float>* %521, align 4, !noalias !133
  %522 = add nuw nsw i64 %353, 104
  %523 = getelementptr float, float* %228, i64 %522
  %524 = getelementptr float, float* %229, i64 %522
  %525 = bitcast float* %523 to <8 x float>*
  %526 = load <8 x float>, <8 x float>* %525, align 4, !noalias !133
  %527 = fadd <8 x float> %526, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %528 = fcmp olt <8 x float> %527, zeroinitializer
  %529 = select <8 x i1> %528, <8 x float> zeroinitializer, <8 x float> %527
  %530 = fcmp ogt <8 x float> %529, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %531 = select <8 x i1> %530, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %529
  %532 = fmul <8 x float> %526, %531
  %533 = fdiv <8 x float> %532, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %534 = bitcast float* %524 to <8 x float>*
  store <8 x float> %533, <8 x float>* %534, align 4, !noalias !133
  %535 = add nuw nsw i64 %351, 1
  %exitcond567.not.i = icmp eq i64 %535, 112
  br i1 %exitcond567.not.i, label %exit104.i, label %cond105.preheader.i

exit104.i:                                        ; preds = %cond105.preheader.i
  %536 = add nuw nsw i64 %241, 1
  %exitcond568.not.i = icmp eq i64 %536, 16
  br i1 %exitcond568.not.i, label %exit.i, label %cond102.preheader.i

cond114.preheader.i:                              ; preds = %exit116.i, %exit.i
  %537 = phi i64 [ 0, %exit.i ], [ %766, %exit116.i ]
  %538 = mul nuw nsw i64 %537, 12544
  br label %cond117.preheader.i

exit113.i:                                        ; preds = %exit116.i
  %539 = alloca [3 x i8*], align 8
  %540 = alloca [3 x i64], align 16
  %541 = alloca [8 x i64], align 8
  %542 = alloca [3 x i8], align 1
  %.sub28.i = getelementptr inbounds [3 x i8], [3 x i8]* %542, i64 0, i64 0
  %.sub27.i = getelementptr inbounds [8 x i64], [8 x i64]* %541, i64 0, i64 0
  %.sub26.i = getelementptr inbounds [3 x i64], [3 x i64]* %540, i64 0, i64 0
  %.sub25.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %539, i64 0, i64 0
  store i8* %malloccall54.i, i8** %.sub25.i, align 8, !noalias !133
  store i8 6, i8* %.sub28.i, align 1, !noalias !133
  %543 = bitcast [8 x i64]* %541 to <4 x i64>*
  store <4 x i64> <i64 1, i64 64, i64 112, i64 112>, <4 x i64>* %543, align 8, !noalias !133
  %544 = getelementptr inbounds [3 x i8*], [3 x i8*]* %539, i64 0, i64 1
  store i8* %malloccall65.i, i8** %544, align 8, !noalias !133
  %545 = getelementptr inbounds [3 x i8], [3 x i8]* %542, i64 0, i64 1
  store i8 6, i8* %545, align 1, !noalias !133
  %546 = bitcast [3 x i64]* %540 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %546, align 16, !noalias !133
  %547 = getelementptr inbounds [8 x i64], [8 x i64]* %541, i64 0, i64 4
  %548 = bitcast i64* %547 to <4 x i64>*
  store <4 x i64> <i64 1, i64 16, i64 112, i64 112>, <4 x i64>* %548, align 8, !noalias !133
  %549 = getelementptr inbounds [3 x i8*], [3 x i8*]* %539, i64 0, i64 2
  %550 = bitcast i8** %549 to float**
  store float* %32, float** %550, align 8, !noalias !133
  %551 = getelementptr inbounds [3 x i8], [3 x i8]* %542, i64 0, i64 2
  store i8 6, i8* %551, align 1, !noalias !133
  %552 = getelementptr inbounds [3 x i64], [3 x i64]* %540, i64 0, i64 2
  store i64 0, i64* %552, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub25.i, i64* nonnull %.sub26.i, i64* nonnull %.sub27.i, i8* nonnull %.sub28.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %553 = alloca [3 x i8*], align 8
  %554 = alloca [3 x i64], align 16
  %555 = alloca [8 x i64], align 8
  %556 = alloca [3 x i8], align 1
  %.sub33.i = getelementptr inbounds [3 x i8], [3 x i8]* %556, i64 0, i64 0
  %.sub32.i = getelementptr inbounds [8 x i64], [8 x i64]* %555, i64 0, i64 0
  %.sub31.i = getelementptr inbounds [3 x i64], [3 x i64]* %554, i64 0, i64 0
  %.sub30.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %553, i64 0, i64 0
  store i8* %malloccall48.i, i8** %.sub30.i, align 8, !noalias !133
  store i8 6, i8* %.sub33.i, align 1, !noalias !133
  %557 = bitcast [8 x i64]* %555 to <4 x i64>*
  store <4 x i64> <i64 1, i64 64, i64 56, i64 56>, <4 x i64>* %557, align 8, !noalias !133
  %558 = getelementptr inbounds [3 x i8*], [3 x i8*]* %553, i64 0, i64 1
  store i8* %malloccall54.i, i8** %558, align 8, !noalias !133
  %559 = getelementptr inbounds [3 x i8], [3 x i8]* %556, i64 0, i64 1
  store i8 6, i8* %559, align 1, !noalias !133
  %560 = bitcast [3 x i64]* %554 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %560, align 16, !noalias !133
  %561 = getelementptr inbounds [8 x i64], [8 x i64]* %555, i64 0, i64 4
  %562 = bitcast i64* %561 to <4 x i64>*
  store <4 x i64> <i64 1, i64 64, i64 112, i64 112>, <4 x i64>* %562, align 8, !noalias !133
  %563 = getelementptr inbounds [3 x i8*], [3 x i8*]* %553, i64 0, i64 2
  %564 = bitcast i8** %563 to float**
  store float* %35, float** %564, align 8, !noalias !133
  %565 = getelementptr inbounds [3 x i8], [3 x i8]* %556, i64 0, i64 2
  store i8 6, i8* %565, align 1, !noalias !133
  %566 = getelementptr inbounds [3 x i64], [3 x i64]* %554, i64 0, i64 2
  store i64 0, i64* %566, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub30.i, i64* nonnull %.sub31.i, i64* nonnull %.sub32.i, i8* nonnull %.sub33.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %567 = alloca [3 x i8*], align 8
  %568 = alloca [3 x i64], align 16
  %569 = alloca [8 x i64], align 8
  %570 = alloca [3 x i8], align 1
  %.sub38.i = getelementptr inbounds [3 x i8], [3 x i8]* %570, i64 0, i64 0
  %.sub37.i = getelementptr inbounds [8 x i64], [8 x i64]* %569, i64 0, i64 0
  %.sub36.i = getelementptr inbounds [3 x i64], [3 x i64]* %568, i64 0, i64 0
  %.sub35.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %567, i64 0, i64 0
  store i8* %malloccall80.i, i8** %.sub35.i, align 8, !noalias !133
  store i8 6, i8* %.sub38.i, align 1, !noalias !133
  %571 = bitcast [8 x i64]* %569 to <4 x i64>*
  store <4 x i64> <i64 1, i64 24, i64 56, i64 56>, <4 x i64>* %571, align 8, !noalias !133
  %572 = getelementptr inbounds [3 x i8*], [3 x i8*]* %567, i64 0, i64 1
  store i8* %malloccall48.i, i8** %572, align 8, !noalias !133
  %573 = getelementptr inbounds [3 x i8], [3 x i8]* %570, i64 0, i64 1
  store i8 6, i8* %573, align 1, !noalias !133
  %574 = bitcast [3 x i64]* %568 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %574, align 16, !noalias !133
  %575 = getelementptr inbounds [8 x i64], [8 x i64]* %569, i64 0, i64 4
  %576 = bitcast i64* %575 to <4 x i64>*
  store <4 x i64> <i64 1, i64 64, i64 56, i64 56>, <4 x i64>* %576, align 8, !noalias !133
  %577 = getelementptr inbounds [3 x i8*], [3 x i8*]* %567, i64 0, i64 2
  %578 = bitcast i8** %577 to float**
  store float* %38, float** %578, align 8, !noalias !133
  %579 = getelementptr inbounds [3 x i8], [3 x i8]* %570, i64 0, i64 2
  store i8 6, i8* %579, align 1, !noalias !133
  %580 = getelementptr inbounds [3 x i64], [3 x i64]* %568, i64 0, i64 2
  store i64 0, i64* %580, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub35.i, i64* nonnull %.sub36.i, i64* nonnull %.sub37.i, i8* nonnull %.sub38.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %581 = alloca [3 x i8*], align 8
  %582 = alloca [3 x i64], align 16
  %583 = alloca [8 x i64], align 8
  %584 = alloca [3 x i8], align 1
  %.sub43.i = getelementptr inbounds [3 x i8], [3 x i8]* %584, i64 0, i64 0
  %.sub42.i = getelementptr inbounds [8 x i64], [8 x i64]* %583, i64 0, i64 0
  %.sub41.i = getelementptr inbounds [3 x i64], [3 x i64]* %582, i64 0, i64 0
  %.sub40.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %581, i64 0, i64 0
  store i8* %malloccall81.i, i8** %.sub40.i, align 8, !noalias !133
  store i8 6, i8* %.sub43.i, align 1, !noalias !133
  %585 = bitcast [8 x i64]* %583 to <4 x i64>*
  store <4 x i64> <i64 1, i64 72, i64 56, i64 56>, <4 x i64>* %585, align 8, !noalias !133
  %586 = getelementptr inbounds [3 x i8*], [3 x i8*]* %581, i64 0, i64 1
  store i8* %malloccall80.i, i8** %586, align 8, !noalias !133
  %587 = getelementptr inbounds [3 x i8], [3 x i8]* %584, i64 0, i64 1
  store i8 6, i8* %587, align 1, !noalias !133
  %588 = bitcast [3 x i64]* %582 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %588, align 16, !noalias !133
  %589 = getelementptr inbounds [8 x i64], [8 x i64]* %583, i64 0, i64 4
  %590 = bitcast i64* %589 to <4 x i64>*
  store <4 x i64> <i64 1, i64 24, i64 56, i64 56>, <4 x i64>* %590, align 8, !noalias !133
  %591 = getelementptr inbounds [3 x i8*], [3 x i8*]* %581, i64 0, i64 2
  %592 = bitcast i8** %591 to float**
  store float* %41, float** %592, align 8, !noalias !133
  %593 = getelementptr inbounds [3 x i8], [3 x i8]* %584, i64 0, i64 2
  store i8 6, i8* %593, align 1, !noalias !133
  %594 = getelementptr inbounds [3 x i64], [3 x i64]* %582, i64 0, i64 2
  store i64 0, i64* %594, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub40.i, i64* nonnull %.sub41.i, i64* nonnull %.sub42.i, i8* nonnull %.sub43.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %595 = alloca [3 x i8*], align 8
  %596 = alloca [3 x i64], align 16
  %597 = alloca [8 x i64], align 8
  %598 = alloca [3 x i8], align 1
  %.sub48.i = getelementptr inbounds [3 x i8], [3 x i8]* %598, i64 0, i64 0
  %.sub47.i = getelementptr inbounds [8 x i64], [8 x i64]* %597, i64 0, i64 0
  %.sub46.i = getelementptr inbounds [3 x i64], [3 x i64]* %596, i64 0, i64 0
  %.sub45.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %595, i64 0, i64 0
  store i8* %malloccall70.i, i8** %.sub45.i, align 8, !noalias !133
  store i8 6, i8* %.sub48.i, align 1, !noalias !133
  %599 = bitcast [8 x i64]* %597 to <4 x i64>*
  store <4 x i64> <i64 1, i64 72, i64 56, i64 56>, <4 x i64>* %599, align 8, !noalias !133
  %600 = getelementptr inbounds [3 x i8*], [3 x i8*]* %595, i64 0, i64 1
  store i8* %malloccall81.i, i8** %600, align 8, !noalias !133
  %601 = getelementptr inbounds [3 x i8], [3 x i8]* %598, i64 0, i64 1
  store i8 6, i8* %601, align 1, !noalias !133
  %602 = bitcast [3 x i64]* %596 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %602, align 16, !noalias !133
  %603 = getelementptr inbounds [8 x i64], [8 x i64]* %597, i64 0, i64 4
  %604 = bitcast i64* %603 to <4 x i64>*
  store <4 x i64> <i64 1, i64 72, i64 56, i64 56>, <4 x i64>* %604, align 8, !noalias !133
  %605 = getelementptr inbounds [3 x i8*], [3 x i8*]* %595, i64 0, i64 2
  %606 = bitcast i8** %605 to float**
  store float* %44, float** %606, align 8, !noalias !133
  %607 = getelementptr inbounds [3 x i8], [3 x i8]* %598, i64 0, i64 2
  store i8 6, i8* %607, align 1, !noalias !133
  %608 = getelementptr inbounds [3 x i64], [3 x i64]* %596, i64 0, i64 2
  store i64 0, i64* %608, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub45.i, i64* nonnull %.sub46.i, i64* nonnull %.sub47.i, i8* nonnull %.sub48.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %609 = alloca [3 x i8*], align 8
  %610 = alloca [3 x i64], align 16
  %611 = alloca [8 x i64], align 8
  %612 = alloca [3 x i8], align 1
  %.sub53.i = getelementptr inbounds [3 x i8], [3 x i8]* %612, i64 0, i64 0
  %.sub52.i = getelementptr inbounds [8 x i64], [8 x i64]* %611, i64 0, i64 0
  %.sub51.i = getelementptr inbounds [3 x i64], [3 x i64]* %610, i64 0, i64 0
  %.sub50.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %609, i64 0, i64 0
  store i8* %malloccall76.i, i8** %.sub50.i, align 8, !noalias !133
  store i8 6, i8* %.sub53.i, align 1, !noalias !133
  %613 = bitcast [8 x i64]* %611 to <4 x i64>*
  store <4 x i64> <i64 1, i64 24, i64 56, i64 56>, <4 x i64>* %613, align 8, !noalias !133
  %614 = getelementptr inbounds [3 x i8*], [3 x i8*]* %609, i64 0, i64 1
  store i8* %malloccall70.i, i8** %614, align 8, !noalias !133
  %615 = getelementptr inbounds [3 x i8], [3 x i8]* %612, i64 0, i64 1
  store i8 6, i8* %615, align 1, !noalias !133
  %616 = bitcast [3 x i64]* %610 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %616, align 16, !noalias !133
  %617 = getelementptr inbounds [8 x i64], [8 x i64]* %611, i64 0, i64 4
  %618 = bitcast i64* %617 to <4 x i64>*
  store <4 x i64> <i64 1, i64 72, i64 56, i64 56>, <4 x i64>* %618, align 8, !noalias !133
  %619 = getelementptr inbounds [3 x i8*], [3 x i8*]* %609, i64 0, i64 2
  %620 = bitcast i8** %619 to float**
  store float* %47, float** %620, align 8, !noalias !133
  %621 = getelementptr inbounds [3 x i8], [3 x i8]* %612, i64 0, i64 2
  store i8 6, i8* %621, align 1, !noalias !133
  %622 = getelementptr inbounds [3 x i64], [3 x i64]* %610, i64 0, i64 2
  store i64 0, i64* %622, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub50.i, i64* nonnull %.sub51.i, i64* nonnull %.sub52.i, i8* nonnull %.sub53.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond126.preheader.i

cond117.preheader.i:                              ; preds = %cond117.preheader.i, %cond114.preheader.i
  %623 = phi i64 [ 0, %cond114.preheader.i ], [ %765, %cond117.preheader.i ]
  %624 = mul nuw nsw i64 %623, 112
  %625 = add nuw nsw i64 %624, %538
  %626 = getelementptr float, float* %295, i64 %625
  %627 = getelementptr float, float* %229, i64 %625
  %628 = getelementptr float, float* %296, i64 %625
  %629 = bitcast float* %626 to <8 x float>*
  %630 = load <8 x float>, <8 x float>* %629, align 4, !noalias !133
  %631 = bitcast float* %627 to <8 x float>*
  %632 = load <8 x float>, <8 x float>* %631, align 4, !noalias !133
  %633 = fadd <8 x float> %630, %632
  %634 = bitcast float* %628 to <8 x float>*
  store <8 x float> %633, <8 x float>* %634, align 4, !noalias !133
  %635 = or i64 %625, 8
  %636 = getelementptr float, float* %295, i64 %635
  %637 = getelementptr float, float* %229, i64 %635
  %638 = getelementptr float, float* %296, i64 %635
  %639 = bitcast float* %636 to <8 x float>*
  %640 = load <8 x float>, <8 x float>* %639, align 4, !noalias !133
  %641 = bitcast float* %637 to <8 x float>*
  %642 = load <8 x float>, <8 x float>* %641, align 4, !noalias !133
  %643 = fadd <8 x float> %640, %642
  %644 = bitcast float* %638 to <8 x float>*
  store <8 x float> %643, <8 x float>* %644, align 4, !noalias !133
  %645 = add nuw nsw i64 %625, 16
  %646 = getelementptr float, float* %295, i64 %645
  %647 = getelementptr float, float* %229, i64 %645
  %648 = getelementptr float, float* %296, i64 %645
  %649 = bitcast float* %646 to <8 x float>*
  %650 = load <8 x float>, <8 x float>* %649, align 4, !noalias !133
  %651 = bitcast float* %647 to <8 x float>*
  %652 = load <8 x float>, <8 x float>* %651, align 4, !noalias !133
  %653 = fadd <8 x float> %650, %652
  %654 = bitcast float* %648 to <8 x float>*
  store <8 x float> %653, <8 x float>* %654, align 4, !noalias !133
  %655 = add nuw nsw i64 %625, 24
  %656 = getelementptr float, float* %295, i64 %655
  %657 = getelementptr float, float* %229, i64 %655
  %658 = getelementptr float, float* %296, i64 %655
  %659 = bitcast float* %656 to <8 x float>*
  %660 = load <8 x float>, <8 x float>* %659, align 4, !noalias !133
  %661 = bitcast float* %657 to <8 x float>*
  %662 = load <8 x float>, <8 x float>* %661, align 4, !noalias !133
  %663 = fadd <8 x float> %660, %662
  %664 = bitcast float* %658 to <8 x float>*
  store <8 x float> %663, <8 x float>* %664, align 4, !noalias !133
  %665 = add nuw nsw i64 %625, 32
  %666 = getelementptr float, float* %295, i64 %665
  %667 = getelementptr float, float* %229, i64 %665
  %668 = getelementptr float, float* %296, i64 %665
  %669 = bitcast float* %666 to <8 x float>*
  %670 = load <8 x float>, <8 x float>* %669, align 4, !noalias !133
  %671 = bitcast float* %667 to <8 x float>*
  %672 = load <8 x float>, <8 x float>* %671, align 4, !noalias !133
  %673 = fadd <8 x float> %670, %672
  %674 = bitcast float* %668 to <8 x float>*
  store <8 x float> %673, <8 x float>* %674, align 4, !noalias !133
  %675 = add nuw nsw i64 %625, 40
  %676 = getelementptr float, float* %295, i64 %675
  %677 = getelementptr float, float* %229, i64 %675
  %678 = getelementptr float, float* %296, i64 %675
  %679 = bitcast float* %676 to <8 x float>*
  %680 = load <8 x float>, <8 x float>* %679, align 4, !noalias !133
  %681 = bitcast float* %677 to <8 x float>*
  %682 = load <8 x float>, <8 x float>* %681, align 4, !noalias !133
  %683 = fadd <8 x float> %680, %682
  %684 = bitcast float* %678 to <8 x float>*
  store <8 x float> %683, <8 x float>* %684, align 4, !noalias !133
  %685 = add nuw nsw i64 %625, 48
  %686 = getelementptr float, float* %295, i64 %685
  %687 = getelementptr float, float* %229, i64 %685
  %688 = getelementptr float, float* %296, i64 %685
  %689 = bitcast float* %686 to <8 x float>*
  %690 = load <8 x float>, <8 x float>* %689, align 4, !noalias !133
  %691 = bitcast float* %687 to <8 x float>*
  %692 = load <8 x float>, <8 x float>* %691, align 4, !noalias !133
  %693 = fadd <8 x float> %690, %692
  %694 = bitcast float* %688 to <8 x float>*
  store <8 x float> %693, <8 x float>* %694, align 4, !noalias !133
  %695 = add nuw nsw i64 %625, 56
  %696 = getelementptr float, float* %295, i64 %695
  %697 = getelementptr float, float* %229, i64 %695
  %698 = getelementptr float, float* %296, i64 %695
  %699 = bitcast float* %696 to <8 x float>*
  %700 = load <8 x float>, <8 x float>* %699, align 4, !noalias !133
  %701 = bitcast float* %697 to <8 x float>*
  %702 = load <8 x float>, <8 x float>* %701, align 4, !noalias !133
  %703 = fadd <8 x float> %700, %702
  %704 = bitcast float* %698 to <8 x float>*
  store <8 x float> %703, <8 x float>* %704, align 4, !noalias !133
  %705 = add nuw nsw i64 %625, 64
  %706 = getelementptr float, float* %295, i64 %705
  %707 = getelementptr float, float* %229, i64 %705
  %708 = getelementptr float, float* %296, i64 %705
  %709 = bitcast float* %706 to <8 x float>*
  %710 = load <8 x float>, <8 x float>* %709, align 4, !noalias !133
  %711 = bitcast float* %707 to <8 x float>*
  %712 = load <8 x float>, <8 x float>* %711, align 4, !noalias !133
  %713 = fadd <8 x float> %710, %712
  %714 = bitcast float* %708 to <8 x float>*
  store <8 x float> %713, <8 x float>* %714, align 4, !noalias !133
  %715 = add nuw nsw i64 %625, 72
  %716 = getelementptr float, float* %295, i64 %715
  %717 = getelementptr float, float* %229, i64 %715
  %718 = getelementptr float, float* %296, i64 %715
  %719 = bitcast float* %716 to <8 x float>*
  %720 = load <8 x float>, <8 x float>* %719, align 4, !noalias !133
  %721 = bitcast float* %717 to <8 x float>*
  %722 = load <8 x float>, <8 x float>* %721, align 4, !noalias !133
  %723 = fadd <8 x float> %720, %722
  %724 = bitcast float* %718 to <8 x float>*
  store <8 x float> %723, <8 x float>* %724, align 4, !noalias !133
  %725 = add nuw nsw i64 %625, 80
  %726 = getelementptr float, float* %295, i64 %725
  %727 = getelementptr float, float* %229, i64 %725
  %728 = getelementptr float, float* %296, i64 %725
  %729 = bitcast float* %726 to <8 x float>*
  %730 = load <8 x float>, <8 x float>* %729, align 4, !noalias !133
  %731 = bitcast float* %727 to <8 x float>*
  %732 = load <8 x float>, <8 x float>* %731, align 4, !noalias !133
  %733 = fadd <8 x float> %730, %732
  %734 = bitcast float* %728 to <8 x float>*
  store <8 x float> %733, <8 x float>* %734, align 4, !noalias !133
  %735 = add nuw nsw i64 %625, 88
  %736 = getelementptr float, float* %295, i64 %735
  %737 = getelementptr float, float* %229, i64 %735
  %738 = getelementptr float, float* %296, i64 %735
  %739 = bitcast float* %736 to <8 x float>*
  %740 = load <8 x float>, <8 x float>* %739, align 4, !noalias !133
  %741 = bitcast float* %737 to <8 x float>*
  %742 = load <8 x float>, <8 x float>* %741, align 4, !noalias !133
  %743 = fadd <8 x float> %740, %742
  %744 = bitcast float* %738 to <8 x float>*
  store <8 x float> %743, <8 x float>* %744, align 4, !noalias !133
  %745 = add nuw nsw i64 %625, 96
  %746 = getelementptr float, float* %295, i64 %745
  %747 = getelementptr float, float* %229, i64 %745
  %748 = getelementptr float, float* %296, i64 %745
  %749 = bitcast float* %746 to <8 x float>*
  %750 = load <8 x float>, <8 x float>* %749, align 4, !noalias !133
  %751 = bitcast float* %747 to <8 x float>*
  %752 = load <8 x float>, <8 x float>* %751, align 4, !noalias !133
  %753 = fadd <8 x float> %750, %752
  %754 = bitcast float* %748 to <8 x float>*
  store <8 x float> %753, <8 x float>* %754, align 4, !noalias !133
  %755 = add nuw nsw i64 %625, 104
  %756 = getelementptr float, float* %295, i64 %755
  %757 = getelementptr float, float* %229, i64 %755
  %758 = getelementptr float, float* %296, i64 %755
  %759 = bitcast float* %756 to <8 x float>*
  %760 = load <8 x float>, <8 x float>* %759, align 4, !noalias !133
  %761 = bitcast float* %757 to <8 x float>*
  %762 = load <8 x float>, <8 x float>* %761, align 4, !noalias !133
  %763 = fadd <8 x float> %760, %762
  %764 = bitcast float* %758 to <8 x float>*
  store <8 x float> %763, <8 x float>* %764, align 4, !noalias !133
  %765 = add nuw nsw i64 %623, 1
  %exitcond563.not.i = icmp eq i64 %765, 112
  br i1 %exitcond563.not.i, label %exit116.i, label %cond117.preheader.i

exit116.i:                                        ; preds = %cond117.preheader.i
  %766 = add nuw nsw i64 %537, 1
  %exitcond564.not.i = icmp eq i64 %766, 16
  br i1 %exitcond564.not.i, label %exit113.i, label %cond114.preheader.i

cond126.preheader.i:                              ; preds = %exit128.i, %exit113.i
  %767 = phi i64 [ 0, %exit113.i ], [ %913, %exit128.i ]
  %768 = mul nuw nsw i64 %767, 3136
  br label %cond129.preheader.i

exit125.i:                                        ; preds = %exit128.i
  %769 = alloca [3 x i8*], align 8
  %770 = alloca [3 x i64], align 16
  %771 = alloca [8 x i64], align 8
  %772 = alloca [3 x i8], align 1
  %.sub58.i = getelementptr inbounds [3 x i8], [3 x i8]* %772, i64 0, i64 0
  %.sub57.i = getelementptr inbounds [8 x i64], [8 x i64]* %771, i64 0, i64 0
  %.sub56.i = getelementptr inbounds [3 x i64], [3 x i64]* %770, i64 0, i64 0
  %.sub55.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %769, i64 0, i64 0
  store i8* %malloccall37.i, i8** %.sub55.i, align 8, !noalias !133
  store i8 6, i8* %.sub58.i, align 1, !noalias !133
  %773 = bitcast [8 x i64]* %771 to <4 x i64>*
  store <4 x i64> <i64 1, i64 72, i64 56, i64 56>, <4 x i64>* %773, align 8, !noalias !133
  %774 = getelementptr inbounds [3 x i8*], [3 x i8*]* %769, i64 0, i64 1
  store i8* %malloccall72.i, i8** %774, align 8, !noalias !133
  %775 = getelementptr inbounds [3 x i8], [3 x i8]* %772, i64 0, i64 1
  store i8 6, i8* %775, align 1, !noalias !133
  %776 = bitcast [3 x i64]* %770 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %776, align 16, !noalias !133
  %777 = getelementptr inbounds [8 x i64], [8 x i64]* %771, i64 0, i64 4
  %778 = bitcast i64* %777 to <4 x i64>*
  store <4 x i64> <i64 1, i64 24, i64 56, i64 56>, <4 x i64>* %778, align 8, !noalias !133
  %779 = getelementptr inbounds [3 x i8*], [3 x i8*]* %769, i64 0, i64 2
  %780 = bitcast i8** %779 to float**
  store float* %50, float** %780, align 8, !noalias !133
  %781 = getelementptr inbounds [3 x i8], [3 x i8]* %772, i64 0, i64 2
  store i8 6, i8* %781, align 1, !noalias !133
  %782 = getelementptr inbounds [3 x i64], [3 x i64]* %770, i64 0, i64 2
  store i64 0, i64* %782, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub55.i, i64* nonnull %.sub56.i, i64* nonnull %.sub57.i, i8* nonnull %.sub58.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %783 = alloca [3 x i8*], align 8
  %784 = alloca [3 x i64], align 16
  %785 = alloca [8 x i64], align 8
  %786 = alloca [3 x i8], align 1
  %.sub63.i = getelementptr inbounds [3 x i8], [3 x i8]* %786, i64 0, i64 0
  %.sub62.i = getelementptr inbounds [8 x i64], [8 x i64]* %785, i64 0, i64 0
  %.sub61.i = getelementptr inbounds [3 x i64], [3 x i64]* %784, i64 0, i64 0
  %.sub60.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %783, i64 0, i64 0
  store i8* %malloccall75.i, i8** %.sub60.i, align 8, !noalias !133
  store i8 6, i8* %.sub63.i, align 1, !noalias !133
  %787 = bitcast [8 x i64]* %785 to <4 x i64>*
  store <4 x i64> <i64 1, i64 72, i64 28, i64 28>, <4 x i64>* %787, align 8, !noalias !133
  %788 = getelementptr inbounds [3 x i8*], [3 x i8*]* %783, i64 0, i64 1
  store i8* %malloccall37.i, i8** %788, align 8, !noalias !133
  %789 = getelementptr inbounds [3 x i8], [3 x i8]* %786, i64 0, i64 1
  store i8 6, i8* %789, align 1, !noalias !133
  %790 = bitcast [3 x i64]* %784 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %790, align 16, !noalias !133
  %791 = getelementptr inbounds [8 x i64], [8 x i64]* %785, i64 0, i64 4
  %792 = bitcast i64* %791 to <4 x i64>*
  store <4 x i64> <i64 1, i64 72, i64 56, i64 56>, <4 x i64>* %792, align 8, !noalias !133
  %793 = getelementptr inbounds [3 x i8*], [3 x i8*]* %783, i64 0, i64 2
  %794 = bitcast i8** %793 to float**
  store float* %53, float** %794, align 8, !noalias !133
  %795 = getelementptr inbounds [3 x i8], [3 x i8]* %786, i64 0, i64 2
  store i8 6, i8* %795, align 1, !noalias !133
  %796 = getelementptr inbounds [3 x i64], [3 x i64]* %784, i64 0, i64 2
  store i64 0, i64* %796, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub60.i, i64* nonnull %.sub61.i, i64* nonnull %.sub62.i, i8* nonnull %.sub63.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %797 = alloca [2 x i8*], align 8
  %798 = alloca <2 x i64>, align 16
  %799 = alloca [8 x i64], align 8
  %800 = alloca [2 x i8], align 1
  %801 = alloca <2 x i64>, align 16
  %.sub69.i = getelementptr inbounds <2 x i64>, <2 x i64>* %801, i64 0, i64 0
  %.sub68.i = getelementptr inbounds [2 x i8], [2 x i8]* %800, i64 0, i64 0
  %.sub67.i = getelementptr inbounds [8 x i64], [8 x i64]* %799, i64 0, i64 0
  %.sub66.i = getelementptr inbounds <2 x i64>, <2 x i64>* %798, i64 0, i64 0
  %.sub65.i = getelementptr inbounds [2 x i8*], [2 x i8*]* %797, i64 0, i64 0
  %802 = bitcast [2 x i8*]* %797 to [288 x float]**
  store [288 x float]* %6, [288 x float]** %802, align 8, !noalias !133
  store i8 6, i8* %.sub68.i, align 1, !noalias !133
  %803 = bitcast [8 x i64]* %799 to <4 x i64>*
  store <4 x i64> <i64 1, i64 72, i64 1, i64 1>, <4 x i64>* %803, align 8, !noalias !133
  %804 = getelementptr inbounds [2 x i8*], [2 x i8*]* %797, i64 0, i64 1
  store i8* %malloccall75.i, i8** %804, align 8, !noalias !133
  %805 = getelementptr inbounds [2 x i8], [2 x i8]* %800, i64 0, i64 1
  store i8 6, i8* %805, align 1, !noalias !133
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %798, align 16, !noalias !133
  %806 = getelementptr inbounds [8 x i64], [8 x i64]* %799, i64 0, i64 4
  %807 = bitcast i64* %806 to <4 x i64>*
  store <4 x i64> <i64 1, i64 72, i64 28, i64 28>, <4 x i64>* %807, align 8, !noalias !133
  store <2 x i64> <i64 1, i64 1>, <2 x i64>* %801, align 16, !noalias !133
  call void @nnc_aten_adaptive_avg_pool2d(i64 2, i8** nonnull %.sub65.i, i64* nonnull %.sub66.i, i64* nonnull %.sub67.i, i8* nonnull %.sub68.i, i64 2, i64* nonnull %.sub69.i) #0, !noalias !135
  %808 = alloca [3 x i8*], align 8
  %809 = alloca [3 x i64], align 16
  %810 = alloca [8 x i64], align 8
  %811 = alloca [3 x i8], align 1
  %.sub73.i = getelementptr inbounds [3 x i8], [3 x i8]* %811, i64 0, i64 0
  %.sub72.i = getelementptr inbounds [8 x i64], [8 x i64]* %810, i64 0, i64 0
  %.sub71.i = getelementptr inbounds [3 x i64], [3 x i64]* %809, i64 0, i64 0
  %.sub70.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %808, i64 0, i64 0
  %812 = bitcast [3 x i8*]* %808 to [96 x float]**
  store [96 x float]* %5, [96 x float]** %812, align 8, !noalias !133
  store i8 6, i8* %.sub73.i, align 1, !noalias !133
  %813 = bitcast [8 x i64]* %810 to <4 x i64>*
  store <4 x i64> <i64 1, i64 24, i64 1, i64 1>, <4 x i64>* %813, align 8, !noalias !133
  %814 = getelementptr inbounds [3 x i8*], [3 x i8*]* %808, i64 0, i64 1
  %815 = bitcast i8** %814 to [288 x float]**
  store [288 x float]* %6, [288 x float]** %815, align 8, !noalias !133
  %816 = getelementptr inbounds [3 x i8], [3 x i8]* %811, i64 0, i64 1
  store i8 6, i8* %816, align 1, !noalias !133
  %817 = bitcast [3 x i64]* %809 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %817, align 16, !noalias !133
  %818 = getelementptr inbounds [8 x i64], [8 x i64]* %810, i64 0, i64 4
  %819 = bitcast i64* %818 to <4 x i64>*
  store <4 x i64> <i64 1, i64 72, i64 1, i64 1>, <4 x i64>* %819, align 8, !noalias !133
  %820 = getelementptr inbounds [3 x i8*], [3 x i8*]* %808, i64 0, i64 2
  %821 = bitcast i8** %820 to float**
  store float* %56, float** %821, align 8, !noalias !133
  %822 = getelementptr inbounds [3 x i8], [3 x i8]* %811, i64 0, i64 2
  store i8 6, i8* %822, align 1, !noalias !133
  %823 = getelementptr inbounds [3 x i64], [3 x i64]* %809, i64 0, i64 2
  store i64 0, i64* %823, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub70.i, i64* nonnull %.sub71.i, i64* nonnull %.sub72.i, i8* nonnull %.sub73.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %824 = alloca [3 x i8*], align 8
  %825 = alloca [3 x i64], align 16
  %826 = alloca [8 x i64], align 8
  %827 = alloca [3 x i8], align 1
  %.sub78.i = getelementptr inbounds [3 x i8], [3 x i8]* %827, i64 0, i64 0
  %.sub77.i = getelementptr inbounds [8 x i64], [8 x i64]* %826, i64 0, i64 0
  %.sub76.i = getelementptr inbounds [3 x i64], [3 x i64]* %825, i64 0, i64 0
  %.sub75.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %824, i64 0, i64 0
  %828 = bitcast [3 x i8*]* %824 to [288 x float]**
  store [288 x float]* %3, [288 x float]** %828, align 8, !noalias !133
  store i8 6, i8* %.sub78.i, align 1, !noalias !133
  %829 = bitcast [8 x i64]* %826 to <4 x i64>*
  store <4 x i64> <i64 1, i64 72, i64 1, i64 1>, <4 x i64>* %829, align 8, !noalias !133
  %830 = getelementptr inbounds [3 x i8*], [3 x i8*]* %824, i64 0, i64 1
  %831 = bitcast i8** %830 to [96 x float]**
  store [96 x float]* %5, [96 x float]** %831, align 8, !noalias !133
  %832 = getelementptr inbounds [3 x i8], [3 x i8]* %827, i64 0, i64 1
  store i8 6, i8* %832, align 1, !noalias !133
  %833 = bitcast [3 x i64]* %825 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %833, align 16, !noalias !133
  %834 = getelementptr inbounds [8 x i64], [8 x i64]* %826, i64 0, i64 4
  %835 = bitcast i64* %834 to <4 x i64>*
  store <4 x i64> <i64 1, i64 24, i64 1, i64 1>, <4 x i64>* %835, align 8, !noalias !133
  %836 = getelementptr inbounds [3 x i8*], [3 x i8*]* %824, i64 0, i64 2
  %837 = bitcast i8** %836 to float**
  store float* %59, float** %837, align 8, !noalias !133
  %838 = getelementptr inbounds [3 x i8], [3 x i8]* %827, i64 0, i64 2
  store i8 6, i8* %838, align 1, !noalias !133
  %839 = getelementptr inbounds [3 x i64], [3 x i64]* %825, i64 0, i64 2
  store i64 0, i64* %839, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub75.i, i64* nonnull %.sub76.i, i64* nonnull %.sub77.i, i8* nonnull %.sub78.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond138.preheader.i

cond129.preheader.i:                              ; preds = %cond129.preheader.i, %cond126.preheader.i
  %840 = phi i64 [ 0, %cond126.preheader.i ], [ %912, %cond129.preheader.i ]
  %841 = mul nuw nsw i64 %840, 56
  %842 = add nuw nsw i64 %841, %768
  %843 = getelementptr float, float* %302, i64 %842
  %844 = getelementptr float, float* %306, i64 %842
  %845 = getelementptr float, float* %298, i64 %842
  %846 = bitcast float* %843 to <8 x float>*
  %847 = load <8 x float>, <8 x float>* %846, align 4, !noalias !133
  %848 = bitcast float* %844 to <8 x float>*
  %849 = load <8 x float>, <8 x float>* %848, align 4, !noalias !133
  %850 = fadd <8 x float> %847, %849
  %851 = bitcast float* %845 to <8 x float>*
  store <8 x float> %850, <8 x float>* %851, align 4, !noalias !133
  %852 = add nuw nsw i64 %842, 8
  %853 = getelementptr float, float* %302, i64 %852
  %854 = getelementptr float, float* %306, i64 %852
  %855 = getelementptr float, float* %298, i64 %852
  %856 = bitcast float* %853 to <8 x float>*
  %857 = load <8 x float>, <8 x float>* %856, align 4, !noalias !133
  %858 = bitcast float* %854 to <8 x float>*
  %859 = load <8 x float>, <8 x float>* %858, align 4, !noalias !133
  %860 = fadd <8 x float> %857, %859
  %861 = bitcast float* %855 to <8 x float>*
  store <8 x float> %860, <8 x float>* %861, align 4, !noalias !133
  %862 = add nuw nsw i64 %842, 16
  %863 = getelementptr float, float* %302, i64 %862
  %864 = getelementptr float, float* %306, i64 %862
  %865 = getelementptr float, float* %298, i64 %862
  %866 = bitcast float* %863 to <8 x float>*
  %867 = load <8 x float>, <8 x float>* %866, align 4, !noalias !133
  %868 = bitcast float* %864 to <8 x float>*
  %869 = load <8 x float>, <8 x float>* %868, align 4, !noalias !133
  %870 = fadd <8 x float> %867, %869
  %871 = bitcast float* %865 to <8 x float>*
  store <8 x float> %870, <8 x float>* %871, align 4, !noalias !133
  %872 = add nuw nsw i64 %842, 24
  %873 = getelementptr float, float* %302, i64 %872
  %874 = getelementptr float, float* %306, i64 %872
  %875 = getelementptr float, float* %298, i64 %872
  %876 = bitcast float* %873 to <8 x float>*
  %877 = load <8 x float>, <8 x float>* %876, align 4, !noalias !133
  %878 = bitcast float* %874 to <8 x float>*
  %879 = load <8 x float>, <8 x float>* %878, align 4, !noalias !133
  %880 = fadd <8 x float> %877, %879
  %881 = bitcast float* %875 to <8 x float>*
  store <8 x float> %880, <8 x float>* %881, align 4, !noalias !133
  %882 = add nuw nsw i64 %842, 32
  %883 = getelementptr float, float* %302, i64 %882
  %884 = getelementptr float, float* %306, i64 %882
  %885 = getelementptr float, float* %298, i64 %882
  %886 = bitcast float* %883 to <8 x float>*
  %887 = load <8 x float>, <8 x float>* %886, align 4, !noalias !133
  %888 = bitcast float* %884 to <8 x float>*
  %889 = load <8 x float>, <8 x float>* %888, align 4, !noalias !133
  %890 = fadd <8 x float> %887, %889
  %891 = bitcast float* %885 to <8 x float>*
  store <8 x float> %890, <8 x float>* %891, align 4, !noalias !133
  %892 = add nuw nsw i64 %842, 40
  %893 = getelementptr float, float* %302, i64 %892
  %894 = getelementptr float, float* %306, i64 %892
  %895 = getelementptr float, float* %298, i64 %892
  %896 = bitcast float* %893 to <8 x float>*
  %897 = load <8 x float>, <8 x float>* %896, align 4, !noalias !133
  %898 = bitcast float* %894 to <8 x float>*
  %899 = load <8 x float>, <8 x float>* %898, align 4, !noalias !133
  %900 = fadd <8 x float> %897, %899
  %901 = bitcast float* %895 to <8 x float>*
  store <8 x float> %900, <8 x float>* %901, align 4, !noalias !133
  %902 = add nuw nsw i64 %842, 48
  %903 = getelementptr float, float* %302, i64 %902
  %904 = getelementptr float, float* %306, i64 %902
  %905 = getelementptr float, float* %298, i64 %902
  %906 = bitcast float* %903 to <8 x float>*
  %907 = load <8 x float>, <8 x float>* %906, align 4, !noalias !133
  %908 = bitcast float* %904 to <8 x float>*
  %909 = load <8 x float>, <8 x float>* %908, align 4, !noalias !133
  %910 = fadd <8 x float> %907, %909
  %911 = bitcast float* %905 to <8 x float>*
  store <8 x float> %910, <8 x float>* %911, align 4, !noalias !133
  %912 = add nuw nsw i64 %840, 1
  %exitcond559.not.i = icmp eq i64 %912, 56
  br i1 %exitcond559.not.i, label %exit128.i, label %cond129.preheader.i

exit128.i:                                        ; preds = %cond129.preheader.i
  %913 = add nuw nsw i64 %767, 1
  %exitcond560.not.i = icmp eq i64 %913, 24
  br i1 %exitcond560.not.i, label %exit125.i, label %cond126.preheader.i

cond138.preheader.i:                              ; preds = %exit140.i, %exit125.i
  %914 = phi i64 [ 0, %exit125.i ], [ %1074, %exit140.i ]
  %915 = mul nuw nsw i64 %914, 784
  %916 = getelementptr [288 x float], [288 x float]* %3, i64 0, i64 %914
  %917 = load float, float* %916, align 4, !noalias !133
  %918 = fadd float %917, 3.000000e+00
  %919 = fcmp olt float %918, 0.000000e+00
  %920 = select i1 %919, float 0.000000e+00, float %918
  %921 = fcmp ogt float %920, 6.000000e+00
  %.op408.i = fdiv float %918, 6.000000e+00
  %.op407.i = select i1 %919, float 0.000000e+00, float %.op408.i
  %922 = select i1 %921, float 1.000000e+00, float %.op407.i
  %923 = add nuw nsw i64 %915, 24
  %924 = insertelement <8 x float> poison, float %922, i32 0
  %925 = shufflevector <8 x float> %924, <8 x float> undef, <8 x i32> zeroinitializer
  %926 = insertelement <4 x float> poison, float %922, i32 0
  %927 = shufflevector <4 x float> %926, <4 x float> undef, <4 x i32> zeroinitializer
  br label %cond141.preheader.i

exit137.i:                                        ; preds = %exit140.i
  %928 = alloca [3 x i8*], align 8
  %929 = alloca [3 x i64], align 16
  %930 = alloca [8 x i64], align 8
  %931 = alloca [3 x i8], align 1
  %.sub83.i = getelementptr inbounds [3 x i8], [3 x i8]* %931, i64 0, i64 0
  %.sub82.i = getelementptr inbounds [8 x i64], [8 x i64]* %930, i64 0, i64 0
  %.sub81.i = getelementptr inbounds [3 x i64], [3 x i64]* %929, i64 0, i64 0
  %.sub80.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %928, i64 0, i64 0
  store i8* %malloccall32.i, i8** %.sub80.i, align 8, !noalias !133
  store i8 6, i8* %.sub83.i, align 1, !noalias !133
  %932 = bitcast [8 x i64]* %930 to <4 x i64>*
  store <4 x i64> <i64 1, i64 40, i64 28, i64 28>, <4 x i64>* %932, align 8, !noalias !133
  %933 = getelementptr inbounds [3 x i8*], [3 x i8*]* %928, i64 0, i64 1
  store i8* %malloccall77.i, i8** %933, align 8, !noalias !133
  %934 = getelementptr inbounds [3 x i8], [3 x i8]* %931, i64 0, i64 1
  store i8 6, i8* %934, align 1, !noalias !133
  %935 = bitcast [3 x i64]* %929 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %935, align 16, !noalias !133
  %936 = getelementptr inbounds [8 x i64], [8 x i64]* %930, i64 0, i64 4
  %937 = bitcast i64* %936 to <4 x i64>*
  store <4 x i64> <i64 1, i64 72, i64 28, i64 28>, <4 x i64>* %937, align 8, !noalias !133
  %938 = getelementptr inbounds [3 x i8*], [3 x i8*]* %928, i64 0, i64 2
  %939 = bitcast i8** %938 to float**
  store float* %62, float** %939, align 8, !noalias !133
  %940 = getelementptr inbounds [3 x i8], [3 x i8]* %931, i64 0, i64 2
  store i8 6, i8* %940, align 1, !noalias !133
  %941 = getelementptr inbounds [3 x i64], [3 x i64]* %929, i64 0, i64 2
  store i64 0, i64* %941, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub80.i, i64* nonnull %.sub81.i, i64* nonnull %.sub82.i, i8* nonnull %.sub83.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %942 = alloca [3 x i8*], align 8
  %943 = alloca [3 x i64], align 16
  %944 = alloca [8 x i64], align 8
  %945 = alloca [3 x i8], align 1
  %.sub88.i = getelementptr inbounds [3 x i8], [3 x i8]* %945, i64 0, i64 0
  %.sub87.i = getelementptr inbounds [8 x i64], [8 x i64]* %944, i64 0, i64 0
  %.sub86.i = getelementptr inbounds [3 x i64], [3 x i64]* %943, i64 0, i64 0
  %.sub85.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %942, i64 0, i64 0
  store i8* %malloccall71.i, i8** %.sub85.i, align 8, !noalias !133
  store i8 6, i8* %.sub88.i, align 1, !noalias !133
  %946 = bitcast [8 x i64]* %944 to <4 x i64>*
  store <4 x i64> <i64 1, i64 120, i64 28, i64 28>, <4 x i64>* %946, align 8, !noalias !133
  %947 = getelementptr inbounds [3 x i8*], [3 x i8*]* %942, i64 0, i64 1
  store i8* %malloccall32.i, i8** %947, align 8, !noalias !133
  %948 = getelementptr inbounds [3 x i8], [3 x i8]* %945, i64 0, i64 1
  store i8 6, i8* %948, align 1, !noalias !133
  %949 = bitcast [3 x i64]* %943 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %949, align 16, !noalias !133
  %950 = getelementptr inbounds [8 x i64], [8 x i64]* %944, i64 0, i64 4
  %951 = bitcast i64* %950 to <4 x i64>*
  store <4 x i64> <i64 1, i64 40, i64 28, i64 28>, <4 x i64>* %951, align 8, !noalias !133
  %952 = getelementptr inbounds [3 x i8*], [3 x i8*]* %942, i64 0, i64 2
  %953 = bitcast i8** %952 to float**
  store float* %65, float** %953, align 8, !noalias !133
  %954 = getelementptr inbounds [3 x i8], [3 x i8]* %945, i64 0, i64 2
  store i8 6, i8* %954, align 1, !noalias !133
  %955 = getelementptr inbounds [3 x i64], [3 x i64]* %943, i64 0, i64 2
  store i64 0, i64* %955, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub85.i, i64* nonnull %.sub86.i, i64* nonnull %.sub87.i, i8* nonnull %.sub88.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %956 = alloca [3 x i8*], align 8
  %957 = alloca [3 x i64], align 16
  %958 = alloca [8 x i64], align 8
  %959 = alloca [3 x i8], align 1
  %.sub93.i = getelementptr inbounds [3 x i8], [3 x i8]* %959, i64 0, i64 0
  %.sub92.i = getelementptr inbounds [8 x i64], [8 x i64]* %958, i64 0, i64 0
  %.sub91.i = getelementptr inbounds [3 x i64], [3 x i64]* %957, i64 0, i64 0
  %.sub90.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %956, i64 0, i64 0
  store i8* %malloccall20.i, i8** %.sub90.i, align 8, !noalias !133
  store i8 6, i8* %.sub93.i, align 1, !noalias !133
  %960 = bitcast [8 x i64]* %958 to <4 x i64>*
  store <4 x i64> <i64 1, i64 120, i64 28, i64 28>, <4 x i64>* %960, align 8, !noalias !133
  %961 = getelementptr inbounds [3 x i8*], [3 x i8*]* %956, i64 0, i64 1
  store i8* %malloccall71.i, i8** %961, align 8, !noalias !133
  %962 = getelementptr inbounds [3 x i8], [3 x i8]* %959, i64 0, i64 1
  store i8 6, i8* %962, align 1, !noalias !133
  %963 = bitcast [3 x i64]* %957 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %963, align 16, !noalias !133
  %964 = getelementptr inbounds [8 x i64], [8 x i64]* %958, i64 0, i64 4
  %965 = bitcast i64* %964 to <4 x i64>*
  store <4 x i64> <i64 1, i64 120, i64 28, i64 28>, <4 x i64>* %965, align 8, !noalias !133
  %966 = getelementptr inbounds [3 x i8*], [3 x i8*]* %956, i64 0, i64 2
  %967 = bitcast i8** %966 to float**
  store float* %68, float** %967, align 8, !noalias !133
  %968 = getelementptr inbounds [3 x i8], [3 x i8]* %959, i64 0, i64 2
  store i8 6, i8* %968, align 1, !noalias !133
  %969 = getelementptr inbounds [3 x i64], [3 x i64]* %957, i64 0, i64 2
  store i64 0, i64* %969, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub90.i, i64* nonnull %.sub91.i, i64* nonnull %.sub92.i, i8* nonnull %.sub93.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %970 = alloca [2 x i8*], align 8
  %971 = alloca <2 x i64>, align 16
  %972 = alloca [8 x i64], align 8
  %973 = alloca [2 x i8], align 1
  %974 = alloca <2 x i64>, align 16
  %.sub99.i = getelementptr inbounds <2 x i64>, <2 x i64>* %974, i64 0, i64 0
  %.sub98.i = getelementptr inbounds [2 x i8], [2 x i8]* %973, i64 0, i64 0
  %.sub97.i = getelementptr inbounds [8 x i64], [8 x i64]* %972, i64 0, i64 0
  %.sub96.i = getelementptr inbounds <2 x i64>, <2 x i64>* %971, i64 0, i64 0
  %.sub95.i = getelementptr inbounds [2 x i8*], [2 x i8*]* %970, i64 0, i64 0
  %975 = bitcast [2 x i8*]* %970 to [480 x float]**
  store [480 x float]* %7, [480 x float]** %975, align 8, !noalias !133
  store i8 6, i8* %.sub98.i, align 1, !noalias !133
  %976 = bitcast [8 x i64]* %972 to <4 x i64>*
  store <4 x i64> <i64 1, i64 120, i64 1, i64 1>, <4 x i64>* %976, align 8, !noalias !133
  %977 = getelementptr inbounds [2 x i8*], [2 x i8*]* %970, i64 0, i64 1
  store i8* %malloccall20.i, i8** %977, align 8, !noalias !133
  %978 = getelementptr inbounds [2 x i8], [2 x i8]* %973, i64 0, i64 1
  store i8 6, i8* %978, align 1, !noalias !133
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %971, align 16, !noalias !133
  %979 = getelementptr inbounds [8 x i64], [8 x i64]* %972, i64 0, i64 4
  %980 = bitcast i64* %979 to <4 x i64>*
  store <4 x i64> <i64 1, i64 120, i64 28, i64 28>, <4 x i64>* %980, align 8, !noalias !133
  store <2 x i64> <i64 1, i64 1>, <2 x i64>* %974, align 16, !noalias !133
  call void @nnc_aten_adaptive_avg_pool2d(i64 2, i8** nonnull %.sub95.i, i64* nonnull %.sub96.i, i64* nonnull %.sub97.i, i8* nonnull %.sub98.i, i64 2, i64* nonnull %.sub99.i) #0, !noalias !135
  %981 = alloca [3 x i8*], align 8
  %982 = alloca [3 x i64], align 16
  %983 = alloca [8 x i64], align 8
  %984 = alloca [3 x i8], align 1
  %.sub103.i = getelementptr inbounds [3 x i8], [3 x i8]* %984, i64 0, i64 0
  %.sub102.i = getelementptr inbounds [8 x i64], [8 x i64]* %983, i64 0, i64 0
  %.sub101.i = getelementptr inbounds [3 x i64], [3 x i64]* %982, i64 0, i64 0
  %.sub100.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %981, i64 0, i64 0
  %985 = bitcast [3 x i8*]* %981 to [128 x float]**
  store [128 x float]* %11, [128 x float]** %985, align 8, !noalias !133
  store i8 6, i8* %.sub103.i, align 1, !noalias !133
  %986 = bitcast [8 x i64]* %983 to <4 x i64>*
  store <4 x i64> <i64 1, i64 32, i64 1, i64 1>, <4 x i64>* %986, align 8, !noalias !133
  %987 = getelementptr inbounds [3 x i8*], [3 x i8*]* %981, i64 0, i64 1
  %988 = bitcast i8** %987 to [480 x float]**
  store [480 x float]* %7, [480 x float]** %988, align 8, !noalias !133
  %989 = getelementptr inbounds [3 x i8], [3 x i8]* %984, i64 0, i64 1
  store i8 6, i8* %989, align 1, !noalias !133
  %990 = bitcast [3 x i64]* %982 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %990, align 16, !noalias !133
  %991 = getelementptr inbounds [8 x i64], [8 x i64]* %983, i64 0, i64 4
  %992 = bitcast i64* %991 to <4 x i64>*
  store <4 x i64> <i64 1, i64 120, i64 1, i64 1>, <4 x i64>* %992, align 8, !noalias !133
  %993 = getelementptr inbounds [3 x i8*], [3 x i8*]* %981, i64 0, i64 2
  %994 = bitcast i8** %993 to float**
  store float* %71, float** %994, align 8, !noalias !133
  %995 = getelementptr inbounds [3 x i8], [3 x i8]* %984, i64 0, i64 2
  store i8 6, i8* %995, align 1, !noalias !133
  %996 = getelementptr inbounds [3 x i64], [3 x i64]* %982, i64 0, i64 2
  store i64 0, i64* %996, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub100.i, i64* nonnull %.sub101.i, i64* nonnull %.sub102.i, i8* nonnull %.sub103.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %997 = alloca [3 x i8*], align 8
  %998 = alloca [3 x i64], align 16
  %999 = alloca [8 x i64], align 8
  %1000 = alloca [3 x i8], align 1
  %.sub108.i = getelementptr inbounds [3 x i8], [3 x i8]* %1000, i64 0, i64 0
  %.sub107.i = getelementptr inbounds [8 x i64], [8 x i64]* %999, i64 0, i64 0
  %.sub106.i = getelementptr inbounds [3 x i64], [3 x i64]* %998, i64 0, i64 0
  %.sub105.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %997, i64 0, i64 0
  %1001 = bitcast [3 x i8*]* %997 to [480 x float]**
  store [480 x float]* %8, [480 x float]** %1001, align 8, !noalias !133
  store i8 6, i8* %.sub108.i, align 1, !noalias !133
  %1002 = bitcast [8 x i64]* %999 to <4 x i64>*
  store <4 x i64> <i64 1, i64 120, i64 1, i64 1>, <4 x i64>* %1002, align 8, !noalias !133
  %1003 = getelementptr inbounds [3 x i8*], [3 x i8*]* %997, i64 0, i64 1
  %1004 = bitcast i8** %1003 to [128 x float]**
  store [128 x float]* %11, [128 x float]** %1004, align 8, !noalias !133
  %1005 = getelementptr inbounds [3 x i8], [3 x i8]* %1000, i64 0, i64 1
  store i8 6, i8* %1005, align 1, !noalias !133
  %1006 = bitcast [3 x i64]* %998 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %1006, align 16, !noalias !133
  %1007 = getelementptr inbounds [8 x i64], [8 x i64]* %999, i64 0, i64 4
  %1008 = bitcast i64* %1007 to <4 x i64>*
  store <4 x i64> <i64 1, i64 32, i64 1, i64 1>, <4 x i64>* %1008, align 8, !noalias !133
  %1009 = getelementptr inbounds [3 x i8*], [3 x i8*]* %997, i64 0, i64 2
  %1010 = bitcast i8** %1009 to float**
  store float* %74, float** %1010, align 8, !noalias !133
  %1011 = getelementptr inbounds [3 x i8], [3 x i8]* %1000, i64 0, i64 2
  store i8 6, i8* %1011, align 1, !noalias !133
  %1012 = getelementptr inbounds [3 x i64], [3 x i64]* %998, i64 0, i64 2
  store i64 0, i64* %1012, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub105.i, i64* nonnull %.sub106.i, i64* nonnull %.sub107.i, i8* nonnull %.sub108.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond153.preheader.i

cond141.preheader.i:                              ; preds = %cond141.preheader.i, %cond138.preheader.i
  %1013 = phi i64 [ 0, %cond138.preheader.i ], [ %1073, %cond141.preheader.i ]
  %1014 = mul nuw nsw i64 %1013, 28
  %1015 = add nuw nsw i64 %1014, %915
  %1016 = getelementptr float, float* %301, i64 %1015
  %1017 = getelementptr float, float* %303, i64 %1015
  %1018 = bitcast float* %1016 to <8 x float>*
  %1019 = load <8 x float>, <8 x float>* %1018, align 4, !noalias !133
  %1020 = fmul <8 x float> %1019, %925
  %1021 = bitcast float* %1017 to <8 x float>*
  store <8 x float> %1020, <8 x float>* %1021, align 4, !noalias !133
  %1022 = add nuw nsw i64 %1015, 8
  %1023 = getelementptr float, float* %301, i64 %1022
  %1024 = getelementptr float, float* %303, i64 %1022
  %1025 = bitcast float* %1023 to <8 x float>*
  %1026 = load <8 x float>, <8 x float>* %1025, align 4, !noalias !133
  %1027 = fmul <8 x float> %1026, %925
  %1028 = bitcast float* %1024 to <8 x float>*
  store <8 x float> %1027, <8 x float>* %1028, align 4, !noalias !133
  %1029 = add nuw nsw i64 %1015, 16
  %1030 = getelementptr float, float* %301, i64 %1029
  %1031 = getelementptr float, float* %303, i64 %1029
  %1032 = bitcast float* %1030 to <8 x float>*
  %1033 = load <8 x float>, <8 x float>* %1032, align 4, !noalias !133
  %1034 = fmul <8 x float> %1033, %925
  %1035 = bitcast float* %1031 to <8 x float>*
  store <8 x float> %1034, <8 x float>* %1035, align 4, !noalias !133
  %1036 = add nuw nsw i64 %923, %1014
  %1037 = getelementptr float, float* %301, i64 %1036
  %1038 = getelementptr float, float* %303, i64 %1036
  %1039 = bitcast float* %1037 to <4 x float>*
  %1040 = load <4 x float>, <4 x float>* %1039, align 4, !noalias !133
  %1041 = fmul <4 x float> %1040, %927
  %1042 = bitcast float* %1038 to <4 x float>*
  store <4 x float> %1041, <4 x float>* %1042, align 4, !noalias !133
  %1043 = or i64 %1013, 1
  %1044 = mul nuw nsw i64 %1043, 28
  %1045 = add nuw nsw i64 %1044, %915
  %1046 = getelementptr float, float* %301, i64 %1045
  %1047 = getelementptr float, float* %303, i64 %1045
  %1048 = bitcast float* %1046 to <8 x float>*
  %1049 = load <8 x float>, <8 x float>* %1048, align 4, !noalias !133
  %1050 = fmul <8 x float> %1049, %925
  %1051 = bitcast float* %1047 to <8 x float>*
  store <8 x float> %1050, <8 x float>* %1051, align 4, !noalias !133
  %1052 = add nuw nsw i64 %1045, 8
  %1053 = getelementptr float, float* %301, i64 %1052
  %1054 = getelementptr float, float* %303, i64 %1052
  %1055 = bitcast float* %1053 to <8 x float>*
  %1056 = load <8 x float>, <8 x float>* %1055, align 4, !noalias !133
  %1057 = fmul <8 x float> %1056, %925
  %1058 = bitcast float* %1054 to <8 x float>*
  store <8 x float> %1057, <8 x float>* %1058, align 4, !noalias !133
  %1059 = add nuw nsw i64 %1045, 16
  %1060 = getelementptr float, float* %301, i64 %1059
  %1061 = getelementptr float, float* %303, i64 %1059
  %1062 = bitcast float* %1060 to <8 x float>*
  %1063 = load <8 x float>, <8 x float>* %1062, align 4, !noalias !133
  %1064 = fmul <8 x float> %1063, %925
  %1065 = bitcast float* %1061 to <8 x float>*
  store <8 x float> %1064, <8 x float>* %1065, align 4, !noalias !133
  %1066 = add nuw nsw i64 %923, %1044
  %1067 = getelementptr float, float* %301, i64 %1066
  %1068 = getelementptr float, float* %303, i64 %1066
  %1069 = bitcast float* %1067 to <4 x float>*
  %1070 = load <4 x float>, <4 x float>* %1069, align 4, !noalias !133
  %1071 = fmul <4 x float> %1070, %927
  %1072 = bitcast float* %1068 to <4 x float>*
  store <4 x float> %1071, <4 x float>* %1072, align 4, !noalias !133
  %1073 = add nuw nsw i64 %1013, 2
  %exitcond555.not.1.i = icmp eq i64 %1073, 28
  br i1 %exitcond555.not.1.i, label %exit140.i, label %cond141.preheader.i

exit140.i:                                        ; preds = %cond141.preheader.i
  %1074 = add nuw nsw i64 %914, 1
  %exitcond556.not.i = icmp eq i64 %1074, 72
  br i1 %exitcond556.not.i, label %exit137.i, label %cond138.preheader.i

cond153.preheader.i:                              ; preds = %exit155.i, %exit137.i
  %1075 = phi i64 [ 0, %exit137.i ], [ %1164, %exit155.i ]
  %1076 = mul nuw nsw i64 %1075, 784
  %1077 = getelementptr [480 x float], [480 x float]* %8, i64 0, i64 %1075
  %1078 = load float, float* %1077, align 4, !noalias !133
  %1079 = fadd float %1078, 3.000000e+00
  %1080 = fcmp olt float %1079, 0.000000e+00
  %1081 = select i1 %1080, float 0.000000e+00, float %1079
  %1082 = fcmp ogt float %1081, 6.000000e+00
  %.op404.i = fdiv float %1079, 6.000000e+00
  %.op403.i = select i1 %1080, float 0.000000e+00, float %.op404.i
  %1083 = select i1 %1082, float 1.000000e+00, float %.op403.i
  %1084 = add nuw nsw i64 %1076, 24
  %1085 = insertelement <8 x float> poison, float %1083, i32 0
  %1086 = shufflevector <8 x float> %1085, <8 x float> undef, <8 x i32> zeroinitializer
  %1087 = insertelement <4 x float> poison, float %1083, i32 0
  %1088 = shufflevector <4 x float> %1087, <4 x float> undef, <4 x i32> zeroinitializer
  br label %cond156.preheader.i

exit152.i:                                        ; preds = %exit155.i
  %1089 = alloca [3 x i8*], align 8
  %1090 = alloca [3 x i64], align 16
  %1091 = alloca [8 x i64], align 8
  %1092 = alloca [3 x i8], align 1
  %.sub113.i = getelementptr inbounds [3 x i8], [3 x i8]* %1092, i64 0, i64 0
  %.sub112.i = getelementptr inbounds [8 x i64], [8 x i64]* %1091, i64 0, i64 0
  %.sub111.i = getelementptr inbounds [3 x i64], [3 x i64]* %1090, i64 0, i64 0
  %.sub110.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %1089, i64 0, i64 0
  store i8* %malloccall46.i, i8** %.sub110.i, align 8, !noalias !133
  store i8 6, i8* %.sub113.i, align 1, !noalias !133
  %1093 = bitcast [8 x i64]* %1091 to <4 x i64>*
  store <4 x i64> <i64 1, i64 40, i64 28, i64 28>, <4 x i64>* %1093, align 8, !noalias !133
  %1094 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1089, i64 0, i64 1
  store i8* %malloccall29.i, i8** %1094, align 8, !noalias !133
  %1095 = getelementptr inbounds [3 x i8], [3 x i8]* %1092, i64 0, i64 1
  store i8 6, i8* %1095, align 1, !noalias !133
  %1096 = bitcast [3 x i64]* %1090 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %1096, align 16, !noalias !133
  %1097 = getelementptr inbounds [8 x i64], [8 x i64]* %1091, i64 0, i64 4
  %1098 = bitcast i64* %1097 to <4 x i64>*
  store <4 x i64> <i64 1, i64 120, i64 28, i64 28>, <4 x i64>* %1098, align 8, !noalias !133
  %1099 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1089, i64 0, i64 2
  %1100 = bitcast i8** %1099 to float**
  store float* %77, float** %1100, align 8, !noalias !133
  %1101 = getelementptr inbounds [3 x i8], [3 x i8]* %1092, i64 0, i64 2
  store i8 6, i8* %1101, align 1, !noalias !133
  %1102 = getelementptr inbounds [3 x i64], [3 x i64]* %1090, i64 0, i64 2
  store i64 0, i64* %1102, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub110.i, i64* nonnull %.sub111.i, i64* nonnull %.sub112.i, i8* nonnull %.sub113.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond168.preheader.i

cond156.preheader.i:                              ; preds = %cond156.preheader.i, %cond153.preheader.i
  %1103 = phi i64 [ 0, %cond153.preheader.i ], [ %1163, %cond156.preheader.i ]
  %1104 = mul nuw nsw i64 %1103, 28
  %1105 = add nuw nsw i64 %1104, %1076
  %1106 = getelementptr float, float* %261, i64 %1105
  %1107 = getelementptr float, float* %269, i64 %1105
  %1108 = bitcast float* %1106 to <8 x float>*
  %1109 = load <8 x float>, <8 x float>* %1108, align 4, !noalias !133
  %1110 = fmul <8 x float> %1109, %1086
  %1111 = bitcast float* %1107 to <8 x float>*
  store <8 x float> %1110, <8 x float>* %1111, align 4, !noalias !133
  %1112 = add nuw nsw i64 %1105, 8
  %1113 = getelementptr float, float* %261, i64 %1112
  %1114 = getelementptr float, float* %269, i64 %1112
  %1115 = bitcast float* %1113 to <8 x float>*
  %1116 = load <8 x float>, <8 x float>* %1115, align 4, !noalias !133
  %1117 = fmul <8 x float> %1116, %1086
  %1118 = bitcast float* %1114 to <8 x float>*
  store <8 x float> %1117, <8 x float>* %1118, align 4, !noalias !133
  %1119 = add nuw nsw i64 %1105, 16
  %1120 = getelementptr float, float* %261, i64 %1119
  %1121 = getelementptr float, float* %269, i64 %1119
  %1122 = bitcast float* %1120 to <8 x float>*
  %1123 = load <8 x float>, <8 x float>* %1122, align 4, !noalias !133
  %1124 = fmul <8 x float> %1123, %1086
  %1125 = bitcast float* %1121 to <8 x float>*
  store <8 x float> %1124, <8 x float>* %1125, align 4, !noalias !133
  %1126 = add nuw nsw i64 %1084, %1104
  %1127 = getelementptr float, float* %261, i64 %1126
  %1128 = getelementptr float, float* %269, i64 %1126
  %1129 = bitcast float* %1127 to <4 x float>*
  %1130 = load <4 x float>, <4 x float>* %1129, align 4, !noalias !133
  %1131 = fmul <4 x float> %1130, %1088
  %1132 = bitcast float* %1128 to <4 x float>*
  store <4 x float> %1131, <4 x float>* %1132, align 4, !noalias !133
  %1133 = or i64 %1103, 1
  %1134 = mul nuw nsw i64 %1133, 28
  %1135 = add nuw nsw i64 %1134, %1076
  %1136 = getelementptr float, float* %261, i64 %1135
  %1137 = getelementptr float, float* %269, i64 %1135
  %1138 = bitcast float* %1136 to <8 x float>*
  %1139 = load <8 x float>, <8 x float>* %1138, align 4, !noalias !133
  %1140 = fmul <8 x float> %1139, %1086
  %1141 = bitcast float* %1137 to <8 x float>*
  store <8 x float> %1140, <8 x float>* %1141, align 4, !noalias !133
  %1142 = add nuw nsw i64 %1135, 8
  %1143 = getelementptr float, float* %261, i64 %1142
  %1144 = getelementptr float, float* %269, i64 %1142
  %1145 = bitcast float* %1143 to <8 x float>*
  %1146 = load <8 x float>, <8 x float>* %1145, align 4, !noalias !133
  %1147 = fmul <8 x float> %1146, %1086
  %1148 = bitcast float* %1144 to <8 x float>*
  store <8 x float> %1147, <8 x float>* %1148, align 4, !noalias !133
  %1149 = add nuw nsw i64 %1135, 16
  %1150 = getelementptr float, float* %261, i64 %1149
  %1151 = getelementptr float, float* %269, i64 %1149
  %1152 = bitcast float* %1150 to <8 x float>*
  %1153 = load <8 x float>, <8 x float>* %1152, align 4, !noalias !133
  %1154 = fmul <8 x float> %1153, %1086
  %1155 = bitcast float* %1151 to <8 x float>*
  store <8 x float> %1154, <8 x float>* %1155, align 4, !noalias !133
  %1156 = add nuw nsw i64 %1084, %1134
  %1157 = getelementptr float, float* %261, i64 %1156
  %1158 = getelementptr float, float* %269, i64 %1156
  %1159 = bitcast float* %1157 to <4 x float>*
  %1160 = load <4 x float>, <4 x float>* %1159, align 4, !noalias !133
  %1161 = fmul <4 x float> %1160, %1088
  %1162 = bitcast float* %1158 to <4 x float>*
  store <4 x float> %1161, <4 x float>* %1162, align 4, !noalias !133
  %1163 = add nuw nsw i64 %1103, 2
  %exitcond550.not.1.i = icmp eq i64 %1163, 28
  br i1 %exitcond550.not.1.i, label %exit155.i, label %cond156.preheader.i

exit155.i:                                        ; preds = %cond156.preheader.i
  %1164 = add nuw nsw i64 %1075, 1
  %exitcond551.not.i = icmp eq i64 %1164, 120
  br i1 %exitcond551.not.i, label %exit152.i, label %cond153.preheader.i

cond168.preheader.i:                              ; preds = %exit170.i, %exit152.i
  %1165 = phi i64 [ 0, %exit152.i ], [ %1324, %exit170.i ]
  %1166 = mul nuw nsw i64 %1165, 784
  %1167 = add nuw nsw i64 %1166, 24
  br label %cond171.preheader.i

exit167.i:                                        ; preds = %exit170.i
  %1168 = alloca [3 x i8*], align 8
  %1169 = alloca [3 x i64], align 16
  %1170 = alloca [8 x i64], align 8
  %1171 = alloca [3 x i8], align 1
  %.sub118.i = getelementptr inbounds [3 x i8], [3 x i8]* %1171, i64 0, i64 0
  %.sub117.i = getelementptr inbounds [8 x i64], [8 x i64]* %1170, i64 0, i64 0
  %.sub116.i = getelementptr inbounds [3 x i64], [3 x i64]* %1169, i64 0, i64 0
  %.sub115.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %1168, i64 0, i64 0
  store i8* %malloccall63.i, i8** %.sub115.i, align 8, !noalias !133
  store i8 6, i8* %.sub118.i, align 1, !noalias !133
  %1172 = bitcast [8 x i64]* %1170 to <4 x i64>*
  store <4 x i64> <i64 1, i64 120, i64 28, i64 28>, <4 x i64>* %1172, align 8, !noalias !133
  %1173 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1168, i64 0, i64 1
  store i8* %malloccall44.i, i8** %1173, align 8, !noalias !133
  %1174 = getelementptr inbounds [3 x i8], [3 x i8]* %1171, i64 0, i64 1
  store i8 6, i8* %1174, align 1, !noalias !133
  %1175 = bitcast [3 x i64]* %1169 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %1175, align 16, !noalias !133
  %1176 = getelementptr inbounds [8 x i64], [8 x i64]* %1170, i64 0, i64 4
  %1177 = bitcast i64* %1176 to <4 x i64>*
  store <4 x i64> <i64 1, i64 40, i64 28, i64 28>, <4 x i64>* %1177, align 8, !noalias !133
  %1178 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1168, i64 0, i64 2
  %1179 = bitcast i8** %1178 to float**
  store float* %80, float** %1179, align 8, !noalias !133
  %1180 = getelementptr inbounds [3 x i8], [3 x i8]* %1171, i64 0, i64 2
  store i8 6, i8* %1180, align 1, !noalias !133
  %1181 = getelementptr inbounds [3 x i64], [3 x i64]* %1169, i64 0, i64 2
  store i64 0, i64* %1181, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub115.i, i64* nonnull %.sub116.i, i64* nonnull %.sub117.i, i8* nonnull %.sub118.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %1182 = alloca [3 x i8*], align 8
  %1183 = alloca [3 x i64], align 16
  %1184 = alloca [8 x i64], align 8
  %1185 = alloca [3 x i8], align 1
  %.sub123.i = getelementptr inbounds [3 x i8], [3 x i8]* %1185, i64 0, i64 0
  %.sub122.i = getelementptr inbounds [8 x i64], [8 x i64]* %1184, i64 0, i64 0
  %.sub121.i = getelementptr inbounds [3 x i64], [3 x i64]* %1183, i64 0, i64 0
  %.sub120.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %1182, i64 0, i64 0
  store i8* %malloccall86.i, i8** %.sub120.i, align 8, !noalias !133
  store i8 6, i8* %.sub123.i, align 1, !noalias !133
  %1186 = bitcast [8 x i64]* %1184 to <4 x i64>*
  store <4 x i64> <i64 1, i64 120, i64 28, i64 28>, <4 x i64>* %1186, align 8, !noalias !133
  %1187 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1182, i64 0, i64 1
  store i8* %malloccall63.i, i8** %1187, align 8, !noalias !133
  %1188 = getelementptr inbounds [3 x i8], [3 x i8]* %1185, i64 0, i64 1
  store i8 6, i8* %1188, align 1, !noalias !133
  %1189 = bitcast [3 x i64]* %1183 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %1189, align 16, !noalias !133
  %1190 = getelementptr inbounds [8 x i64], [8 x i64]* %1184, i64 0, i64 4
  %1191 = bitcast i64* %1190 to <4 x i64>*
  store <4 x i64> <i64 1, i64 120, i64 28, i64 28>, <4 x i64>* %1191, align 8, !noalias !133
  %1192 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1182, i64 0, i64 2
  %1193 = bitcast i8** %1192 to float**
  store float* %83, float** %1193, align 8, !noalias !133
  %1194 = getelementptr inbounds [3 x i8], [3 x i8]* %1185, i64 0, i64 2
  store i8 6, i8* %1194, align 1, !noalias !133
  %1195 = getelementptr inbounds [3 x i64], [3 x i64]* %1183, i64 0, i64 2
  store i64 0, i64* %1195, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub120.i, i64* nonnull %.sub121.i, i64* nonnull %.sub122.i, i8* nonnull %.sub123.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %1196 = alloca [2 x i8*], align 8
  %1197 = alloca <2 x i64>, align 16
  %1198 = alloca [8 x i64], align 8
  %1199 = alloca [2 x i8], align 1
  %1200 = alloca <2 x i64>, align 16
  %.sub129.i = getelementptr inbounds <2 x i64>, <2 x i64>* %1200, i64 0, i64 0
  %.sub128.i = getelementptr inbounds [2 x i8], [2 x i8]* %1199, i64 0, i64 0
  %.sub127.i = getelementptr inbounds [8 x i64], [8 x i64]* %1198, i64 0, i64 0
  %.sub126.i = getelementptr inbounds <2 x i64>, <2 x i64>* %1197, i64 0, i64 0
  %.sub125.i = getelementptr inbounds [2 x i8*], [2 x i8*]* %1196, i64 0, i64 0
  %1201 = bitcast [2 x i8*]* %1196 to [480 x float]**
  store [480 x float]* %4, [480 x float]** %1201, align 8, !noalias !133
  store i8 6, i8* %.sub128.i, align 1, !noalias !133
  %1202 = bitcast [8 x i64]* %1198 to <4 x i64>*
  store <4 x i64> <i64 1, i64 120, i64 1, i64 1>, <4 x i64>* %1202, align 8, !noalias !133
  %1203 = getelementptr inbounds [2 x i8*], [2 x i8*]* %1196, i64 0, i64 1
  store i8* %malloccall86.i, i8** %1203, align 8, !noalias !133
  %1204 = getelementptr inbounds [2 x i8], [2 x i8]* %1199, i64 0, i64 1
  store i8 6, i8* %1204, align 1, !noalias !133
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %1197, align 16, !noalias !133
  %1205 = getelementptr inbounds [8 x i64], [8 x i64]* %1198, i64 0, i64 4
  %1206 = bitcast i64* %1205 to <4 x i64>*
  store <4 x i64> <i64 1, i64 120, i64 28, i64 28>, <4 x i64>* %1206, align 8, !noalias !133
  store <2 x i64> <i64 1, i64 1>, <2 x i64>* %1200, align 16, !noalias !133
  call void @nnc_aten_adaptive_avg_pool2d(i64 2, i8** nonnull %.sub125.i, i64* nonnull %.sub126.i, i64* nonnull %.sub127.i, i8* nonnull %.sub128.i, i64 2, i64* nonnull %.sub129.i) #0, !noalias !135
  %1207 = alloca [3 x i8*], align 8
  %1208 = alloca [3 x i64], align 16
  %1209 = alloca [8 x i64], align 8
  %1210 = alloca [3 x i8], align 1
  %.sub133.i = getelementptr inbounds [3 x i8], [3 x i8]* %1210, i64 0, i64 0
  %.sub132.i = getelementptr inbounds [8 x i64], [8 x i64]* %1209, i64 0, i64 0
  %.sub131.i = getelementptr inbounds [3 x i64], [3 x i64]* %1208, i64 0, i64 0
  %.sub130.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %1207, i64 0, i64 0
  %1211 = bitcast [3 x i8*]* %1207 to [128 x float]**
  store [128 x float]* %9, [128 x float]** %1211, align 8, !noalias !133
  store i8 6, i8* %.sub133.i, align 1, !noalias !133
  %1212 = bitcast [8 x i64]* %1209 to <4 x i64>*
  store <4 x i64> <i64 1, i64 32, i64 1, i64 1>, <4 x i64>* %1212, align 8, !noalias !133
  %1213 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1207, i64 0, i64 1
  %1214 = bitcast i8** %1213 to [480 x float]**
  store [480 x float]* %4, [480 x float]** %1214, align 8, !noalias !133
  %1215 = getelementptr inbounds [3 x i8], [3 x i8]* %1210, i64 0, i64 1
  store i8 6, i8* %1215, align 1, !noalias !133
  %1216 = bitcast [3 x i64]* %1208 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %1216, align 16, !noalias !133
  %1217 = getelementptr inbounds [8 x i64], [8 x i64]* %1209, i64 0, i64 4
  %1218 = bitcast i64* %1217 to <4 x i64>*
  store <4 x i64> <i64 1, i64 120, i64 1, i64 1>, <4 x i64>* %1218, align 8, !noalias !133
  %1219 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1207, i64 0, i64 2
  %1220 = bitcast i8** %1219 to float**
  store float* %86, float** %1220, align 8, !noalias !133
  %1221 = getelementptr inbounds [3 x i8], [3 x i8]* %1210, i64 0, i64 2
  store i8 6, i8* %1221, align 1, !noalias !133
  %1222 = getelementptr inbounds [3 x i64], [3 x i64]* %1208, i64 0, i64 2
  store i64 0, i64* %1222, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub130.i, i64* nonnull %.sub131.i, i64* nonnull %.sub132.i, i8* nonnull %.sub133.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %1223 = alloca [3 x i8*], align 8
  %1224 = alloca [3 x i64], align 16
  %1225 = alloca [8 x i64], align 8
  %1226 = alloca [3 x i8], align 1
  %.sub138.i = getelementptr inbounds [3 x i8], [3 x i8]* %1226, i64 0, i64 0
  %.sub137.i = getelementptr inbounds [8 x i64], [8 x i64]* %1225, i64 0, i64 0
  %.sub136.i = getelementptr inbounds [3 x i64], [3 x i64]* %1224, i64 0, i64 0
  %.sub135.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %1223, i64 0, i64 0
  %1227 = bitcast [3 x i8*]* %1223 to [480 x float]**
  store [480 x float]* %10, [480 x float]** %1227, align 8, !noalias !133
  store i8 6, i8* %.sub138.i, align 1, !noalias !133
  %1228 = bitcast [8 x i64]* %1225 to <4 x i64>*
  store <4 x i64> <i64 1, i64 120, i64 1, i64 1>, <4 x i64>* %1228, align 8, !noalias !133
  %1229 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1223, i64 0, i64 1
  %1230 = bitcast i8** %1229 to [128 x float]**
  store [128 x float]* %9, [128 x float]** %1230, align 8, !noalias !133
  %1231 = getelementptr inbounds [3 x i8], [3 x i8]* %1226, i64 0, i64 1
  store i8 6, i8* %1231, align 1, !noalias !133
  %1232 = bitcast [3 x i64]* %1224 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %1232, align 16, !noalias !133
  %1233 = getelementptr inbounds [8 x i64], [8 x i64]* %1225, i64 0, i64 4
  %1234 = bitcast i64* %1233 to <4 x i64>*
  store <4 x i64> <i64 1, i64 32, i64 1, i64 1>, <4 x i64>* %1234, align 8, !noalias !133
  %1235 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1223, i64 0, i64 2
  %1236 = bitcast i8** %1235 to float**
  store float* %89, float** %1236, align 8, !noalias !133
  %1237 = getelementptr inbounds [3 x i8], [3 x i8]* %1226, i64 0, i64 2
  store i8 6, i8* %1237, align 1, !noalias !133
  %1238 = getelementptr inbounds [3 x i64], [3 x i64]* %1224, i64 0, i64 2
  store i64 0, i64* %1238, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub135.i, i64* nonnull %.sub136.i, i64* nonnull %.sub137.i, i8* nonnull %.sub138.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond183.preheader.i

cond171.preheader.i:                              ; preds = %cond171.preheader.i, %cond168.preheader.i
  %1239 = phi i64 [ 0, %cond168.preheader.i ], [ %1323, %cond171.preheader.i ]
  %1240 = mul nuw nsw i64 %1239, 28
  %1241 = add nuw nsw i64 %1240, %1166
  %1242 = getelementptr float, float* %282, i64 %1241
  %1243 = getelementptr float, float* %272, i64 %1241
  %1244 = getelementptr float, float* %280, i64 %1241
  %1245 = bitcast float* %1242 to <8 x float>*
  %1246 = load <8 x float>, <8 x float>* %1245, align 4, !noalias !133
  %1247 = bitcast float* %1243 to <8 x float>*
  %1248 = load <8 x float>, <8 x float>* %1247, align 4, !noalias !133
  %1249 = fadd <8 x float> %1246, %1248
  %1250 = bitcast float* %1244 to <8 x float>*
  store <8 x float> %1249, <8 x float>* %1250, align 4, !noalias !133
  %1251 = add nuw nsw i64 %1241, 8
  %1252 = getelementptr float, float* %282, i64 %1251
  %1253 = getelementptr float, float* %272, i64 %1251
  %1254 = getelementptr float, float* %280, i64 %1251
  %1255 = bitcast float* %1252 to <8 x float>*
  %1256 = load <8 x float>, <8 x float>* %1255, align 4, !noalias !133
  %1257 = bitcast float* %1253 to <8 x float>*
  %1258 = load <8 x float>, <8 x float>* %1257, align 4, !noalias !133
  %1259 = fadd <8 x float> %1256, %1258
  %1260 = bitcast float* %1254 to <8 x float>*
  store <8 x float> %1259, <8 x float>* %1260, align 4, !noalias !133
  %1261 = add nuw nsw i64 %1241, 16
  %1262 = getelementptr float, float* %282, i64 %1261
  %1263 = getelementptr float, float* %272, i64 %1261
  %1264 = getelementptr float, float* %280, i64 %1261
  %1265 = bitcast float* %1262 to <8 x float>*
  %1266 = load <8 x float>, <8 x float>* %1265, align 4, !noalias !133
  %1267 = bitcast float* %1263 to <8 x float>*
  %1268 = load <8 x float>, <8 x float>* %1267, align 4, !noalias !133
  %1269 = fadd <8 x float> %1266, %1268
  %1270 = bitcast float* %1264 to <8 x float>*
  store <8 x float> %1269, <8 x float>* %1270, align 4, !noalias !133
  %1271 = add nuw nsw i64 %1167, %1240
  %1272 = getelementptr float, float* %282, i64 %1271
  %1273 = getelementptr float, float* %272, i64 %1271
  %1274 = getelementptr float, float* %280, i64 %1271
  %1275 = bitcast float* %1272 to <4 x float>*
  %1276 = load <4 x float>, <4 x float>* %1275, align 4, !noalias !133
  %1277 = bitcast float* %1273 to <4 x float>*
  %1278 = load <4 x float>, <4 x float>* %1277, align 4, !noalias !133
  %1279 = fadd <4 x float> %1276, %1278
  %1280 = bitcast float* %1274 to <4 x float>*
  store <4 x float> %1279, <4 x float>* %1280, align 4, !noalias !133
  %1281 = or i64 %1239, 1
  %1282 = mul nuw nsw i64 %1281, 28
  %1283 = add nuw nsw i64 %1282, %1166
  %1284 = getelementptr float, float* %282, i64 %1283
  %1285 = getelementptr float, float* %272, i64 %1283
  %1286 = getelementptr float, float* %280, i64 %1283
  %1287 = bitcast float* %1284 to <8 x float>*
  %1288 = load <8 x float>, <8 x float>* %1287, align 4, !noalias !133
  %1289 = bitcast float* %1285 to <8 x float>*
  %1290 = load <8 x float>, <8 x float>* %1289, align 4, !noalias !133
  %1291 = fadd <8 x float> %1288, %1290
  %1292 = bitcast float* %1286 to <8 x float>*
  store <8 x float> %1291, <8 x float>* %1292, align 4, !noalias !133
  %1293 = add nuw nsw i64 %1283, 8
  %1294 = getelementptr float, float* %282, i64 %1293
  %1295 = getelementptr float, float* %272, i64 %1293
  %1296 = getelementptr float, float* %280, i64 %1293
  %1297 = bitcast float* %1294 to <8 x float>*
  %1298 = load <8 x float>, <8 x float>* %1297, align 4, !noalias !133
  %1299 = bitcast float* %1295 to <8 x float>*
  %1300 = load <8 x float>, <8 x float>* %1299, align 4, !noalias !133
  %1301 = fadd <8 x float> %1298, %1300
  %1302 = bitcast float* %1296 to <8 x float>*
  store <8 x float> %1301, <8 x float>* %1302, align 4, !noalias !133
  %1303 = add nuw nsw i64 %1283, 16
  %1304 = getelementptr float, float* %282, i64 %1303
  %1305 = getelementptr float, float* %272, i64 %1303
  %1306 = getelementptr float, float* %280, i64 %1303
  %1307 = bitcast float* %1304 to <8 x float>*
  %1308 = load <8 x float>, <8 x float>* %1307, align 4, !noalias !133
  %1309 = bitcast float* %1305 to <8 x float>*
  %1310 = load <8 x float>, <8 x float>* %1309, align 4, !noalias !133
  %1311 = fadd <8 x float> %1308, %1310
  %1312 = bitcast float* %1306 to <8 x float>*
  store <8 x float> %1311, <8 x float>* %1312, align 4, !noalias !133
  %1313 = add nuw nsw i64 %1167, %1282
  %1314 = getelementptr float, float* %282, i64 %1313
  %1315 = getelementptr float, float* %272, i64 %1313
  %1316 = getelementptr float, float* %280, i64 %1313
  %1317 = bitcast float* %1314 to <4 x float>*
  %1318 = load <4 x float>, <4 x float>* %1317, align 4, !noalias !133
  %1319 = bitcast float* %1315 to <4 x float>*
  %1320 = load <4 x float>, <4 x float>* %1319, align 4, !noalias !133
  %1321 = fadd <4 x float> %1318, %1320
  %1322 = bitcast float* %1316 to <4 x float>*
  store <4 x float> %1321, <4 x float>* %1322, align 4, !noalias !133
  %1323 = add nuw nsw i64 %1239, 2
  %exitcond545.not.1.i = icmp eq i64 %1323, 28
  br i1 %exitcond545.not.1.i, label %exit170.i, label %cond171.preheader.i

exit170.i:                                        ; preds = %cond171.preheader.i
  %1324 = add nuw nsw i64 %1165, 1
  %exitcond546.not.i = icmp eq i64 %1324, 40
  br i1 %exitcond546.not.i, label %exit167.i, label %cond168.preheader.i

cond183.preheader.i:                              ; preds = %exit185.i, %exit167.i
  %1325 = phi i64 [ 0, %exit167.i ], [ %1414, %exit185.i ]
  %1326 = mul nuw nsw i64 %1325, 784
  %1327 = getelementptr [480 x float], [480 x float]* %10, i64 0, i64 %1325
  %1328 = load float, float* %1327, align 4, !noalias !133
  %1329 = fadd float %1328, 3.000000e+00
  %1330 = fcmp olt float %1329, 0.000000e+00
  %1331 = select i1 %1330, float 0.000000e+00, float %1329
  %1332 = fcmp ogt float %1331, 6.000000e+00
  %.op400.i = fdiv float %1329, 6.000000e+00
  %.op399.i = select i1 %1330, float 0.000000e+00, float %.op400.i
  %1333 = select i1 %1332, float 1.000000e+00, float %.op399.i
  %1334 = add nuw nsw i64 %1326, 24
  %1335 = insertelement <8 x float> poison, float %1333, i32 0
  %1336 = shufflevector <8 x float> %1335, <8 x float> undef, <8 x i32> zeroinitializer
  %1337 = insertelement <4 x float> poison, float %1333, i32 0
  %1338 = shufflevector <4 x float> %1337, <4 x float> undef, <4 x i32> zeroinitializer
  br label %cond186.preheader.i

exit182.i:                                        ; preds = %exit185.i
  %1339 = alloca [3 x i8*], align 8
  %1340 = alloca [3 x i64], align 16
  %1341 = alloca [8 x i64], align 8
  %1342 = alloca [3 x i8], align 1
  %.sub143.i = getelementptr inbounds [3 x i8], [3 x i8]* %1342, i64 0, i64 0
  %.sub142.i = getelementptr inbounds [8 x i64], [8 x i64]* %1341, i64 0, i64 0
  %.sub141.i = getelementptr inbounds [3 x i64], [3 x i64]* %1340, i64 0, i64 0
  %.sub140.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %1339, i64 0, i64 0
  store i8* %malloccall57.i, i8** %.sub140.i, align 8, !noalias !133
  store i8 6, i8* %.sub143.i, align 1, !noalias !133
  %1343 = bitcast [8 x i64]* %1341 to <4 x i64>*
  store <4 x i64> <i64 1, i64 40, i64 28, i64 28>, <4 x i64>* %1343, align 8, !noalias !133
  %1344 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1339, i64 0, i64 1
  store i8* %malloccall87.i, i8** %1344, align 8, !noalias !133
  %1345 = getelementptr inbounds [3 x i8], [3 x i8]* %1342, i64 0, i64 1
  store i8 6, i8* %1345, align 1, !noalias !133
  %1346 = bitcast [3 x i64]* %1340 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %1346, align 16, !noalias !133
  %1347 = getelementptr inbounds [8 x i64], [8 x i64]* %1341, i64 0, i64 4
  %1348 = bitcast i64* %1347 to <4 x i64>*
  store <4 x i64> <i64 1, i64 120, i64 28, i64 28>, <4 x i64>* %1348, align 8, !noalias !133
  %1349 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1339, i64 0, i64 2
  %1350 = bitcast i8** %1349 to float**
  store float* %92, float** %1350, align 8, !noalias !133
  %1351 = getelementptr inbounds [3 x i8], [3 x i8]* %1342, i64 0, i64 2
  store i8 6, i8* %1351, align 1, !noalias !133
  %1352 = getelementptr inbounds [3 x i64], [3 x i64]* %1340, i64 0, i64 2
  store i64 0, i64* %1352, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub140.i, i64* nonnull %.sub141.i, i64* nonnull %.sub142.i, i8* nonnull %.sub143.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond198.preheader.i

cond186.preheader.i:                              ; preds = %cond186.preheader.i, %cond183.preheader.i
  %1353 = phi i64 [ 0, %cond183.preheader.i ], [ %1413, %cond186.preheader.i ]
  %1354 = mul nuw nsw i64 %1353, 28
  %1355 = add nuw nsw i64 %1354, %1326
  %1356 = getelementptr float, float* %311, i64 %1355
  %1357 = getelementptr float, float* %312, i64 %1355
  %1358 = bitcast float* %1356 to <8 x float>*
  %1359 = load <8 x float>, <8 x float>* %1358, align 4, !noalias !133
  %1360 = fmul <8 x float> %1359, %1336
  %1361 = bitcast float* %1357 to <8 x float>*
  store <8 x float> %1360, <8 x float>* %1361, align 4, !noalias !133
  %1362 = add nuw nsw i64 %1355, 8
  %1363 = getelementptr float, float* %311, i64 %1362
  %1364 = getelementptr float, float* %312, i64 %1362
  %1365 = bitcast float* %1363 to <8 x float>*
  %1366 = load <8 x float>, <8 x float>* %1365, align 4, !noalias !133
  %1367 = fmul <8 x float> %1366, %1336
  %1368 = bitcast float* %1364 to <8 x float>*
  store <8 x float> %1367, <8 x float>* %1368, align 4, !noalias !133
  %1369 = add nuw nsw i64 %1355, 16
  %1370 = getelementptr float, float* %311, i64 %1369
  %1371 = getelementptr float, float* %312, i64 %1369
  %1372 = bitcast float* %1370 to <8 x float>*
  %1373 = load <8 x float>, <8 x float>* %1372, align 4, !noalias !133
  %1374 = fmul <8 x float> %1373, %1336
  %1375 = bitcast float* %1371 to <8 x float>*
  store <8 x float> %1374, <8 x float>* %1375, align 4, !noalias !133
  %1376 = add nuw nsw i64 %1334, %1354
  %1377 = getelementptr float, float* %311, i64 %1376
  %1378 = getelementptr float, float* %312, i64 %1376
  %1379 = bitcast float* %1377 to <4 x float>*
  %1380 = load <4 x float>, <4 x float>* %1379, align 4, !noalias !133
  %1381 = fmul <4 x float> %1380, %1338
  %1382 = bitcast float* %1378 to <4 x float>*
  store <4 x float> %1381, <4 x float>* %1382, align 4, !noalias !133
  %1383 = or i64 %1353, 1
  %1384 = mul nuw nsw i64 %1383, 28
  %1385 = add nuw nsw i64 %1384, %1326
  %1386 = getelementptr float, float* %311, i64 %1385
  %1387 = getelementptr float, float* %312, i64 %1385
  %1388 = bitcast float* %1386 to <8 x float>*
  %1389 = load <8 x float>, <8 x float>* %1388, align 4, !noalias !133
  %1390 = fmul <8 x float> %1389, %1336
  %1391 = bitcast float* %1387 to <8 x float>*
  store <8 x float> %1390, <8 x float>* %1391, align 4, !noalias !133
  %1392 = add nuw nsw i64 %1385, 8
  %1393 = getelementptr float, float* %311, i64 %1392
  %1394 = getelementptr float, float* %312, i64 %1392
  %1395 = bitcast float* %1393 to <8 x float>*
  %1396 = load <8 x float>, <8 x float>* %1395, align 4, !noalias !133
  %1397 = fmul <8 x float> %1396, %1336
  %1398 = bitcast float* %1394 to <8 x float>*
  store <8 x float> %1397, <8 x float>* %1398, align 4, !noalias !133
  %1399 = add nuw nsw i64 %1385, 16
  %1400 = getelementptr float, float* %311, i64 %1399
  %1401 = getelementptr float, float* %312, i64 %1399
  %1402 = bitcast float* %1400 to <8 x float>*
  %1403 = load <8 x float>, <8 x float>* %1402, align 4, !noalias !133
  %1404 = fmul <8 x float> %1403, %1336
  %1405 = bitcast float* %1401 to <8 x float>*
  store <8 x float> %1404, <8 x float>* %1405, align 4, !noalias !133
  %1406 = add nuw nsw i64 %1334, %1384
  %1407 = getelementptr float, float* %311, i64 %1406
  %1408 = getelementptr float, float* %312, i64 %1406
  %1409 = bitcast float* %1407 to <4 x float>*
  %1410 = load <4 x float>, <4 x float>* %1409, align 4, !noalias !133
  %1411 = fmul <4 x float> %1410, %1338
  %1412 = bitcast float* %1408 to <4 x float>*
  store <4 x float> %1411, <4 x float>* %1412, align 4, !noalias !133
  %1413 = add nuw nsw i64 %1353, 2
  %exitcond540.not.1.i = icmp eq i64 %1413, 28
  br i1 %exitcond540.not.1.i, label %exit185.i, label %cond186.preheader.i

exit185.i:                                        ; preds = %cond186.preheader.i
  %1414 = add nuw nsw i64 %1325, 1
  %exitcond541.not.i = icmp eq i64 %1414, 120
  br i1 %exitcond541.not.i, label %exit182.i, label %cond183.preheader.i

cond198.preheader.i:                              ; preds = %exit200.i, %exit182.i
  %1415 = phi i64 [ 0, %exit182.i ], [ %1517, %exit200.i ]
  %1416 = mul nuw nsw i64 %1415, 784
  %1417 = add nuw nsw i64 %1416, 24
  br label %cond201.preheader.i

exit197.i:                                        ; preds = %exit200.i
  %1418 = alloca [3 x i8*], align 8
  %1419 = alloca [3 x i64], align 16
  %1420 = alloca [8 x i64], align 8
  %1421 = alloca [3 x i8], align 1
  %.sub148.i = getelementptr inbounds [3 x i8], [3 x i8]* %1421, i64 0, i64 0
  %.sub147.i = getelementptr inbounds [8 x i64], [8 x i64]* %1420, i64 0, i64 0
  %.sub146.i = getelementptr inbounds [3 x i64], [3 x i64]* %1419, i64 0, i64 0
  %.sub145.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %1418, i64 0, i64 0
  store i8* %malloccall52.i, i8** %.sub145.i, align 8, !noalias !133
  store i8 6, i8* %.sub148.i, align 1, !noalias !133
  %1422 = bitcast [8 x i64]* %1420 to <4 x i64>*
  store <4 x i64> <i64 1, i64 240, i64 28, i64 28>, <4 x i64>* %1422, align 8, !noalias !133
  %1423 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1418, i64 0, i64 1
  store i8* %malloccall78.i, i8** %1423, align 8, !noalias !133
  %1424 = getelementptr inbounds [3 x i8], [3 x i8]* %1421, i64 0, i64 1
  store i8 6, i8* %1424, align 1, !noalias !133
  %1425 = bitcast [3 x i64]* %1419 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %1425, align 16, !noalias !133
  %1426 = getelementptr inbounds [8 x i64], [8 x i64]* %1420, i64 0, i64 4
  %1427 = bitcast i64* %1426 to <4 x i64>*
  store <4 x i64> <i64 1, i64 40, i64 28, i64 28>, <4 x i64>* %1427, align 8, !noalias !133
  %1428 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1418, i64 0, i64 2
  %1429 = bitcast i8** %1428 to float**
  store float* %95, float** %1429, align 8, !noalias !133
  %1430 = getelementptr inbounds [3 x i8], [3 x i8]* %1421, i64 0, i64 2
  store i8 6, i8* %1430, align 1, !noalias !133
  %1431 = getelementptr inbounds [3 x i64], [3 x i64]* %1419, i64 0, i64 2
  store i64 0, i64* %1431, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub145.i, i64* nonnull %.sub146.i, i64* nonnull %.sub147.i, i8* nonnull %.sub148.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond213.preheader.i

cond201.preheader.i:                              ; preds = %cond201.preheader.i, %cond198.preheader.i
  %1432 = phi i64 [ 0, %cond198.preheader.i ], [ %1516, %cond201.preheader.i ]
  %1433 = mul nuw nsw i64 %1432, 28
  %1434 = add nuw nsw i64 %1433, %1416
  %1435 = getelementptr float, float* %290, i64 %1434
  %1436 = getelementptr float, float* %280, i64 %1434
  %1437 = getelementptr float, float* %304, i64 %1434
  %1438 = bitcast float* %1435 to <8 x float>*
  %1439 = load <8 x float>, <8 x float>* %1438, align 4, !noalias !133
  %1440 = bitcast float* %1436 to <8 x float>*
  %1441 = load <8 x float>, <8 x float>* %1440, align 4, !noalias !133
  %1442 = fadd <8 x float> %1439, %1441
  %1443 = bitcast float* %1437 to <8 x float>*
  store <8 x float> %1442, <8 x float>* %1443, align 4, !noalias !133
  %1444 = add nuw nsw i64 %1434, 8
  %1445 = getelementptr float, float* %290, i64 %1444
  %1446 = getelementptr float, float* %280, i64 %1444
  %1447 = getelementptr float, float* %304, i64 %1444
  %1448 = bitcast float* %1445 to <8 x float>*
  %1449 = load <8 x float>, <8 x float>* %1448, align 4, !noalias !133
  %1450 = bitcast float* %1446 to <8 x float>*
  %1451 = load <8 x float>, <8 x float>* %1450, align 4, !noalias !133
  %1452 = fadd <8 x float> %1449, %1451
  %1453 = bitcast float* %1447 to <8 x float>*
  store <8 x float> %1452, <8 x float>* %1453, align 4, !noalias !133
  %1454 = add nuw nsw i64 %1434, 16
  %1455 = getelementptr float, float* %290, i64 %1454
  %1456 = getelementptr float, float* %280, i64 %1454
  %1457 = getelementptr float, float* %304, i64 %1454
  %1458 = bitcast float* %1455 to <8 x float>*
  %1459 = load <8 x float>, <8 x float>* %1458, align 4, !noalias !133
  %1460 = bitcast float* %1456 to <8 x float>*
  %1461 = load <8 x float>, <8 x float>* %1460, align 4, !noalias !133
  %1462 = fadd <8 x float> %1459, %1461
  %1463 = bitcast float* %1457 to <8 x float>*
  store <8 x float> %1462, <8 x float>* %1463, align 4, !noalias !133
  %1464 = add nuw nsw i64 %1417, %1433
  %1465 = getelementptr float, float* %290, i64 %1464
  %1466 = getelementptr float, float* %280, i64 %1464
  %1467 = getelementptr float, float* %304, i64 %1464
  %1468 = bitcast float* %1465 to <4 x float>*
  %1469 = load <4 x float>, <4 x float>* %1468, align 4, !noalias !133
  %1470 = bitcast float* %1466 to <4 x float>*
  %1471 = load <4 x float>, <4 x float>* %1470, align 4, !noalias !133
  %1472 = fadd <4 x float> %1469, %1471
  %1473 = bitcast float* %1467 to <4 x float>*
  store <4 x float> %1472, <4 x float>* %1473, align 4, !noalias !133
  %1474 = or i64 %1432, 1
  %1475 = mul nuw nsw i64 %1474, 28
  %1476 = add nuw nsw i64 %1475, %1416
  %1477 = getelementptr float, float* %290, i64 %1476
  %1478 = getelementptr float, float* %280, i64 %1476
  %1479 = getelementptr float, float* %304, i64 %1476
  %1480 = bitcast float* %1477 to <8 x float>*
  %1481 = load <8 x float>, <8 x float>* %1480, align 4, !noalias !133
  %1482 = bitcast float* %1478 to <8 x float>*
  %1483 = load <8 x float>, <8 x float>* %1482, align 4, !noalias !133
  %1484 = fadd <8 x float> %1481, %1483
  %1485 = bitcast float* %1479 to <8 x float>*
  store <8 x float> %1484, <8 x float>* %1485, align 4, !noalias !133
  %1486 = add nuw nsw i64 %1476, 8
  %1487 = getelementptr float, float* %290, i64 %1486
  %1488 = getelementptr float, float* %280, i64 %1486
  %1489 = getelementptr float, float* %304, i64 %1486
  %1490 = bitcast float* %1487 to <8 x float>*
  %1491 = load <8 x float>, <8 x float>* %1490, align 4, !noalias !133
  %1492 = bitcast float* %1488 to <8 x float>*
  %1493 = load <8 x float>, <8 x float>* %1492, align 4, !noalias !133
  %1494 = fadd <8 x float> %1491, %1493
  %1495 = bitcast float* %1489 to <8 x float>*
  store <8 x float> %1494, <8 x float>* %1495, align 4, !noalias !133
  %1496 = add nuw nsw i64 %1476, 16
  %1497 = getelementptr float, float* %290, i64 %1496
  %1498 = getelementptr float, float* %280, i64 %1496
  %1499 = getelementptr float, float* %304, i64 %1496
  %1500 = bitcast float* %1497 to <8 x float>*
  %1501 = load <8 x float>, <8 x float>* %1500, align 4, !noalias !133
  %1502 = bitcast float* %1498 to <8 x float>*
  %1503 = load <8 x float>, <8 x float>* %1502, align 4, !noalias !133
  %1504 = fadd <8 x float> %1501, %1503
  %1505 = bitcast float* %1499 to <8 x float>*
  store <8 x float> %1504, <8 x float>* %1505, align 4, !noalias !133
  %1506 = add nuw nsw i64 %1417, %1475
  %1507 = getelementptr float, float* %290, i64 %1506
  %1508 = getelementptr float, float* %280, i64 %1506
  %1509 = getelementptr float, float* %304, i64 %1506
  %1510 = bitcast float* %1507 to <4 x float>*
  %1511 = load <4 x float>, <4 x float>* %1510, align 4, !noalias !133
  %1512 = bitcast float* %1508 to <4 x float>*
  %1513 = load <4 x float>, <4 x float>* %1512, align 4, !noalias !133
  %1514 = fadd <4 x float> %1511, %1513
  %1515 = bitcast float* %1509 to <4 x float>*
  store <4 x float> %1514, <4 x float>* %1515, align 4, !noalias !133
  %1516 = add nuw nsw i64 %1432, 2
  %exitcond535.not.1.i = icmp eq i64 %1516, 28
  br i1 %exitcond535.not.1.i, label %exit200.i, label %cond201.preheader.i

exit200.i:                                        ; preds = %cond201.preheader.i
  %1517 = add nuw nsw i64 %1415, 1
  %exitcond536.not.i = icmp eq i64 %1517, 40
  br i1 %exitcond536.not.i, label %exit197.i, label %cond198.preheader.i

cond213.preheader.i:                              ; preds = %exit215.i, %exit197.i
  %1518 = phi i64 [ 0, %exit197.i ], [ %1590, %exit215.i ]
  %1519 = mul nuw nsw i64 %1518, 784
  %1520 = add nuw nsw i64 %1519, 24
  br label %cond216.preheader.i

exit212.i:                                        ; preds = %exit215.i
  %1521 = alloca [3 x i8*], align 8
  %1522 = alloca [3 x i64], align 16
  %1523 = alloca [8 x i64], align 8
  %1524 = alloca [3 x i8], align 1
  %.sub153.i = getelementptr inbounds [3 x i8], [3 x i8]* %1524, i64 0, i64 0
  %.sub152.i = getelementptr inbounds [8 x i64], [8 x i64]* %1523, i64 0, i64 0
  %.sub151.i = getelementptr inbounds [3 x i64], [3 x i64]* %1522, i64 0, i64 0
  %.sub150.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %1521, i64 0, i64 0
  store i8* %malloccall42.i, i8** %.sub150.i, align 8, !noalias !133
  store i8 6, i8* %.sub153.i, align 1, !noalias !133
  %1525 = bitcast [8 x i64]* %1523 to <4 x i64>*
  store <4 x i64> <i64 1, i64 240, i64 14, i64 14>, <4 x i64>* %1525, align 8, !noalias !133
  %1526 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1521, i64 0, i64 1
  store i8* %malloccall73.i, i8** %1526, align 8, !noalias !133
  %1527 = getelementptr inbounds [3 x i8], [3 x i8]* %1524, i64 0, i64 1
  store i8 6, i8* %1527, align 1, !noalias !133
  %1528 = bitcast [3 x i64]* %1522 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %1528, align 16, !noalias !133
  %1529 = getelementptr inbounds [8 x i64], [8 x i64]* %1523, i64 0, i64 4
  %1530 = bitcast i64* %1529 to <4 x i64>*
  store <4 x i64> <i64 1, i64 240, i64 28, i64 28>, <4 x i64>* %1530, align 8, !noalias !133
  %1531 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1521, i64 0, i64 2
  %1532 = bitcast i8** %1531 to float**
  store float* %98, float** %1532, align 8, !noalias !133
  %1533 = getelementptr inbounds [3 x i8], [3 x i8]* %1524, i64 0, i64 2
  store i8 6, i8* %1533, align 1, !noalias !133
  %1534 = getelementptr inbounds [3 x i64], [3 x i64]* %1522, i64 0, i64 2
  store i64 0, i64* %1534, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub150.i, i64* nonnull %.sub151.i, i64* nonnull %.sub152.i, i8* nonnull %.sub153.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond228.preheader.i

cond216.preheader.i:                              ; preds = %cond216.preheader.i, %cond213.preheader.i
  %1535 = phi i64 [ 0, %cond213.preheader.i ], [ %1589, %cond216.preheader.i ]
  %1536 = mul nuw nsw i64 %1535, 28
  %1537 = add nuw nsw i64 %1536, %1519
  %1538 = getelementptr float, float* %286, i64 %1537
  %1539 = getelementptr float, float* %299, i64 %1537
  %1540 = bitcast float* %1538 to <8 x float>*
  %1541 = load <8 x float>, <8 x float>* %1540, align 4, !noalias !133
  %1542 = fadd <8 x float> %1541, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %1543 = fcmp olt <8 x float> %1542, zeroinitializer
  %1544 = select <8 x i1> %1543, <8 x float> zeroinitializer, <8 x float> %1542
  %1545 = fcmp ogt <8 x float> %1544, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1546 = select <8 x i1> %1545, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %1544
  %1547 = fmul <8 x float> %1541, %1546
  %1548 = fdiv <8 x float> %1547, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1549 = bitcast float* %1539 to <8 x float>*
  store <8 x float> %1548, <8 x float>* %1549, align 4, !noalias !133
  %1550 = add nuw nsw i64 %1537, 8
  %1551 = getelementptr float, float* %286, i64 %1550
  %1552 = getelementptr float, float* %299, i64 %1550
  %1553 = bitcast float* %1551 to <8 x float>*
  %1554 = load <8 x float>, <8 x float>* %1553, align 4, !noalias !133
  %1555 = fadd <8 x float> %1554, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %1556 = fcmp olt <8 x float> %1555, zeroinitializer
  %1557 = select <8 x i1> %1556, <8 x float> zeroinitializer, <8 x float> %1555
  %1558 = fcmp ogt <8 x float> %1557, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1559 = select <8 x i1> %1558, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %1557
  %1560 = fmul <8 x float> %1554, %1559
  %1561 = fdiv <8 x float> %1560, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1562 = bitcast float* %1552 to <8 x float>*
  store <8 x float> %1561, <8 x float>* %1562, align 4, !noalias !133
  %1563 = add nuw nsw i64 %1537, 16
  %1564 = getelementptr float, float* %286, i64 %1563
  %1565 = getelementptr float, float* %299, i64 %1563
  %1566 = bitcast float* %1564 to <8 x float>*
  %1567 = load <8 x float>, <8 x float>* %1566, align 4, !noalias !133
  %1568 = fadd <8 x float> %1567, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %1569 = fcmp olt <8 x float> %1568, zeroinitializer
  %1570 = select <8 x i1> %1569, <8 x float> zeroinitializer, <8 x float> %1568
  %1571 = fcmp ogt <8 x float> %1570, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1572 = select <8 x i1> %1571, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %1570
  %1573 = fmul <8 x float> %1567, %1572
  %1574 = fdiv <8 x float> %1573, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1575 = bitcast float* %1565 to <8 x float>*
  store <8 x float> %1574, <8 x float>* %1575, align 4, !noalias !133
  %1576 = add nuw nsw i64 %1520, %1536
  %1577 = getelementptr float, float* %286, i64 %1576
  %1578 = getelementptr float, float* %299, i64 %1576
  %1579 = bitcast float* %1577 to <4 x float>*
  %1580 = load <4 x float>, <4 x float>* %1579, align 4, !noalias !133
  %1581 = fadd <4 x float> %1580, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %1582 = fcmp olt <4 x float> %1581, zeroinitializer
  %1583 = select <4 x i1> %1582, <4 x float> zeroinitializer, <4 x float> %1581
  %1584 = fcmp ogt <4 x float> %1583, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1585 = select <4 x i1> %1584, <4 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <4 x float> %1583
  %1586 = fmul <4 x float> %1580, %1585
  %1587 = fdiv <4 x float> %1586, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1588 = bitcast float* %1578 to <4 x float>*
  store <4 x float> %1587, <4 x float>* %1588, align 4, !noalias !133
  %1589 = add nuw nsw i64 %1535, 1
  %exitcond530.not.i = icmp eq i64 %1589, 28
  br i1 %exitcond530.not.i, label %exit215.i, label %cond216.preheader.i

exit215.i:                                        ; preds = %cond216.preheader.i
  %1590 = add nuw nsw i64 %1518, 1
  %exitcond531.not.i = icmp eq i64 %1590, 240
  br i1 %exitcond531.not.i, label %exit212.i, label %cond213.preheader.i

cond228.preheader.i:                              ; preds = %exit230.i, %exit212.i
  %1591 = phi i64 [ 0, %exit212.i ], [ %1674, %exit230.i ]
  %1592 = mul nuw nsw i64 %1591, 196
  %1593 = add nuw nsw i64 %1592, 8
  %1594 = add nuw nsw i64 %1592, 12
  br label %cond231.preheader.i

exit227.i:                                        ; preds = %exit230.i
  %1595 = alloca [3 x i8*], align 8
  %1596 = alloca [3 x i64], align 16
  %1597 = alloca [8 x i64], align 8
  %1598 = alloca [3 x i8], align 1
  %.sub158.i = getelementptr inbounds [3 x i8], [3 x i8]* %1598, i64 0, i64 0
  %.sub157.i = getelementptr inbounds [8 x i64], [8 x i64]* %1597, i64 0, i64 0
  %.sub156.i = getelementptr inbounds [3 x i64], [3 x i64]* %1596, i64 0, i64 0
  %.sub155.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %1595, i64 0, i64 0
  store i8* %malloccall35.i, i8** %.sub155.i, align 8, !noalias !133
  store i8 6, i8* %.sub158.i, align 1, !noalias !133
  %1599 = bitcast [8 x i64]* %1597 to <4 x i64>*
  store <4 x i64> <i64 1, i64 80, i64 14, i64 14>, <4 x i64>* %1599, align 8, !noalias !133
  %1600 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1595, i64 0, i64 1
  store i8* %malloccall74.i, i8** %1600, align 8, !noalias !133
  %1601 = getelementptr inbounds [3 x i8], [3 x i8]* %1598, i64 0, i64 1
  store i8 6, i8* %1601, align 1, !noalias !133
  %1602 = bitcast [3 x i64]* %1596 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %1602, align 16, !noalias !133
  %1603 = getelementptr inbounds [8 x i64], [8 x i64]* %1597, i64 0, i64 4
  %1604 = bitcast i64* %1603 to <4 x i64>*
  store <4 x i64> <i64 1, i64 240, i64 14, i64 14>, <4 x i64>* %1604, align 8, !noalias !133
  %1605 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1595, i64 0, i64 2
  %1606 = bitcast i8** %1605 to float**
  store float* %101, float** %1606, align 8, !noalias !133
  %1607 = getelementptr inbounds [3 x i8], [3 x i8]* %1598, i64 0, i64 2
  store i8 6, i8* %1607, align 1, !noalias !133
  %1608 = getelementptr inbounds [3 x i64], [3 x i64]* %1596, i64 0, i64 2
  store i64 0, i64* %1608, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub155.i, i64* nonnull %.sub156.i, i64* nonnull %.sub157.i, i8* nonnull %.sub158.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %1609 = alloca [3 x i8*], align 8
  %1610 = alloca [3 x i64], align 16
  %1611 = alloca [8 x i64], align 8
  %1612 = alloca [3 x i8], align 1
  %.sub163.i = getelementptr inbounds [3 x i8], [3 x i8]* %1612, i64 0, i64 0
  %.sub162.i = getelementptr inbounds [8 x i64], [8 x i64]* %1611, i64 0, i64 0
  %.sub161.i = getelementptr inbounds [3 x i64], [3 x i64]* %1610, i64 0, i64 0
  %.sub160.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %1609, i64 0, i64 0
  store i8* %malloccall36.i, i8** %.sub160.i, align 8, !noalias !133
  store i8 6, i8* %.sub163.i, align 1, !noalias !133
  %1613 = bitcast [8 x i64]* %1611 to <4 x i64>*
  store <4 x i64> <i64 1, i64 200, i64 14, i64 14>, <4 x i64>* %1613, align 8, !noalias !133
  %1614 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1609, i64 0, i64 1
  store i8* %malloccall35.i, i8** %1614, align 8, !noalias !133
  %1615 = getelementptr inbounds [3 x i8], [3 x i8]* %1612, i64 0, i64 1
  store i8 6, i8* %1615, align 1, !noalias !133
  %1616 = bitcast [3 x i64]* %1610 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %1616, align 16, !noalias !133
  %1617 = getelementptr inbounds [8 x i64], [8 x i64]* %1611, i64 0, i64 4
  %1618 = bitcast i64* %1617 to <4 x i64>*
  store <4 x i64> <i64 1, i64 80, i64 14, i64 14>, <4 x i64>* %1618, align 8, !noalias !133
  %1619 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1609, i64 0, i64 2
  %1620 = bitcast i8** %1619 to float**
  store float* %104, float** %1620, align 8, !noalias !133
  %1621 = getelementptr inbounds [3 x i8], [3 x i8]* %1612, i64 0, i64 2
  store i8 6, i8* %1621, align 1, !noalias !133
  %1622 = getelementptr inbounds [3 x i64], [3 x i64]* %1610, i64 0, i64 2
  store i64 0, i64* %1622, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub160.i, i64* nonnull %.sub161.i, i64* nonnull %.sub162.i, i8* nonnull %.sub163.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond243.preheader.i

cond231.preheader.i:                              ; preds = %cond231.preheader.i, %cond228.preheader.i
  %1623 = phi i64 [ 0, %cond228.preheader.i ], [ %1673, %cond231.preheader.i ]
  %1624 = mul nuw nsw i64 %1623, 14
  %1625 = add nuw nsw i64 %1624, %1592
  %1626 = getelementptr float, float* %278, i64 %1625
  %1627 = getelementptr float, float* %300, i64 %1625
  %1628 = bitcast float* %1626 to <8 x float>*
  %1629 = load <8 x float>, <8 x float>* %1628, align 4, !noalias !133
  %1630 = fadd <8 x float> %1629, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %1631 = fcmp olt <8 x float> %1630, zeroinitializer
  %1632 = select <8 x i1> %1631, <8 x float> zeroinitializer, <8 x float> %1630
  %1633 = fcmp ogt <8 x float> %1632, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1634 = select <8 x i1> %1633, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %1632
  %1635 = fmul <8 x float> %1629, %1634
  %1636 = fdiv <8 x float> %1635, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1637 = bitcast float* %1627 to <8 x float>*
  store <8 x float> %1636, <8 x float>* %1637, align 4, !noalias !133
  %1638 = add nuw nsw i64 %1593, %1624
  %1639 = getelementptr float, float* %278, i64 %1638
  %1640 = getelementptr float, float* %300, i64 %1638
  %1641 = bitcast float* %1639 to <4 x float>*
  %1642 = load <4 x float>, <4 x float>* %1641, align 4, !noalias !133
  %1643 = fadd <4 x float> %1642, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %1644 = fcmp olt <4 x float> %1643, zeroinitializer
  %1645 = select <4 x i1> %1644, <4 x float> zeroinitializer, <4 x float> %1643
  %1646 = fcmp ogt <4 x float> %1645, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1647 = select <4 x i1> %1646, <4 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <4 x float> %1645
  %1648 = fmul <4 x float> %1642, %1647
  %1649 = fdiv <4 x float> %1648, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1650 = bitcast float* %1640 to <4 x float>*
  store <4 x float> %1649, <4 x float>* %1650, align 4, !noalias !133
  %1651 = add nuw nsw i64 %1594, %1624
  %1652 = getelementptr float, float* %278, i64 %1651
  %1653 = load float, float* %1652, align 4, !noalias !133
  %1654 = fadd float %1653, 3.000000e+00
  %1655 = fcmp olt float %1654, 0.000000e+00
  %1656 = select i1 %1655, float 0.000000e+00, float %1654
  %1657 = fcmp ogt float %1656, 6.000000e+00
  %1658 = select i1 %1657, float 6.000000e+00, float %1656
  %1659 = fmul float %1653, %1658
  %1660 = fdiv float %1659, 6.000000e+00
  %1661 = getelementptr float, float* %300, i64 %1651
  store float %1660, float* %1661, align 4, !noalias !133
  %1662 = or i64 %1651, 1
  %1663 = getelementptr float, float* %278, i64 %1662
  %1664 = load float, float* %1663, align 4, !noalias !133
  %1665 = fadd float %1664, 3.000000e+00
  %1666 = fcmp olt float %1665, 0.000000e+00
  %1667 = select i1 %1666, float 0.000000e+00, float %1665
  %1668 = fcmp ogt float %1667, 6.000000e+00
  %1669 = select i1 %1668, float 6.000000e+00, float %1667
  %1670 = fmul float %1664, %1669
  %1671 = fdiv float %1670, 6.000000e+00
  %1672 = getelementptr float, float* %300, i64 %1662
  store float %1671, float* %1672, align 4, !noalias !133
  %1673 = add nuw nsw i64 %1623, 1
  %exitcond525.not.i = icmp eq i64 %1673, 14
  br i1 %exitcond525.not.i, label %exit230.i, label %cond231.preheader.i

exit230.i:                                        ; preds = %cond231.preheader.i
  %1674 = add nuw nsw i64 %1591, 1
  %exitcond526.not.i = icmp eq i64 %1674, 240
  br i1 %exitcond526.not.i, label %exit227.i, label %cond228.preheader.i

cond243.preheader.i:                              ; preds = %exit245.i, %exit227.i
  %1675 = phi i64 [ 0, %exit227.i ], [ %1744, %exit245.i ]
  %1676 = mul nuw nsw i64 %1675, 196
  %1677 = add nuw nsw i64 %1676, 8
  %1678 = add nuw nsw i64 %1676, 12
  br label %cond246.preheader.i

exit242.i:                                        ; preds = %exit245.i
  %1679 = alloca [3 x i8*], align 8
  %1680 = alloca [3 x i64], align 16
  %1681 = alloca [8 x i64], align 8
  %1682 = alloca [3 x i8], align 1
  %.sub168.i = getelementptr inbounds [3 x i8], [3 x i8]* %1682, i64 0, i64 0
  %.sub167.i = getelementptr inbounds [8 x i64], [8 x i64]* %1681, i64 0, i64 0
  %.sub166.i = getelementptr inbounds [3 x i64], [3 x i64]* %1680, i64 0, i64 0
  %.sub165.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %1679, i64 0, i64 0
  store i8* %malloccall16.i, i8** %.sub165.i, align 8, !noalias !133
  store i8 6, i8* %.sub168.i, align 1, !noalias !133
  %1683 = bitcast [8 x i64]* %1681 to <4 x i64>*
  store <4 x i64> <i64 1, i64 200, i64 14, i64 14>, <4 x i64>* %1683, align 8, !noalias !133
  %1684 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1679, i64 0, i64 1
  store i8* %malloccall39.i, i8** %1684, align 8, !noalias !133
  %1685 = getelementptr inbounds [3 x i8], [3 x i8]* %1682, i64 0, i64 1
  store i8 6, i8* %1685, align 1, !noalias !133
  %1686 = bitcast [3 x i64]* %1680 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %1686, align 16, !noalias !133
  %1687 = getelementptr inbounds [8 x i64], [8 x i64]* %1681, i64 0, i64 4
  %1688 = bitcast i64* %1687 to <4 x i64>*
  store <4 x i64> <i64 1, i64 200, i64 14, i64 14>, <4 x i64>* %1688, align 8, !noalias !133
  %1689 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1679, i64 0, i64 2
  %1690 = bitcast i8** %1689 to float**
  store float* %107, float** %1690, align 8, !noalias !133
  %1691 = getelementptr inbounds [3 x i8], [3 x i8]* %1682, i64 0, i64 2
  store i8 6, i8* %1691, align 1, !noalias !133
  %1692 = getelementptr inbounds [3 x i64], [3 x i64]* %1680, i64 0, i64 2
  store i64 0, i64* %1692, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub165.i, i64* nonnull %.sub166.i, i64* nonnull %.sub167.i, i8* nonnull %.sub168.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond258.preheader.i

cond246.preheader.i:                              ; preds = %cond246.preheader.i, %cond243.preheader.i
  %1693 = phi i64 [ 0, %cond243.preheader.i ], [ %1743, %cond246.preheader.i ]
  %1694 = mul nuw nsw i64 %1693, 14
  %1695 = add nuw nsw i64 %1694, %1676
  %1696 = getelementptr float, float* %275, i64 %1695
  %1697 = getelementptr float, float* %277, i64 %1695
  %1698 = bitcast float* %1696 to <8 x float>*
  %1699 = load <8 x float>, <8 x float>* %1698, align 4, !noalias !133
  %1700 = fadd <8 x float> %1699, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %1701 = fcmp olt <8 x float> %1700, zeroinitializer
  %1702 = select <8 x i1> %1701, <8 x float> zeroinitializer, <8 x float> %1700
  %1703 = fcmp ogt <8 x float> %1702, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1704 = select <8 x i1> %1703, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %1702
  %1705 = fmul <8 x float> %1699, %1704
  %1706 = fdiv <8 x float> %1705, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1707 = bitcast float* %1697 to <8 x float>*
  store <8 x float> %1706, <8 x float>* %1707, align 4, !noalias !133
  %1708 = add nuw nsw i64 %1677, %1694
  %1709 = getelementptr float, float* %275, i64 %1708
  %1710 = getelementptr float, float* %277, i64 %1708
  %1711 = bitcast float* %1709 to <4 x float>*
  %1712 = load <4 x float>, <4 x float>* %1711, align 4, !noalias !133
  %1713 = fadd <4 x float> %1712, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %1714 = fcmp olt <4 x float> %1713, zeroinitializer
  %1715 = select <4 x i1> %1714, <4 x float> zeroinitializer, <4 x float> %1713
  %1716 = fcmp ogt <4 x float> %1715, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1717 = select <4 x i1> %1716, <4 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <4 x float> %1715
  %1718 = fmul <4 x float> %1712, %1717
  %1719 = fdiv <4 x float> %1718, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1720 = bitcast float* %1710 to <4 x float>*
  store <4 x float> %1719, <4 x float>* %1720, align 4, !noalias !133
  %1721 = add nuw nsw i64 %1678, %1694
  %1722 = getelementptr float, float* %275, i64 %1721
  %1723 = load float, float* %1722, align 4, !noalias !133
  %1724 = fadd float %1723, 3.000000e+00
  %1725 = fcmp olt float %1724, 0.000000e+00
  %1726 = select i1 %1725, float 0.000000e+00, float %1724
  %1727 = fcmp ogt float %1726, 6.000000e+00
  %1728 = select i1 %1727, float 6.000000e+00, float %1726
  %1729 = fmul float %1723, %1728
  %1730 = fdiv float %1729, 6.000000e+00
  %1731 = getelementptr float, float* %277, i64 %1721
  store float %1730, float* %1731, align 4, !noalias !133
  %1732 = or i64 %1721, 1
  %1733 = getelementptr float, float* %275, i64 %1732
  %1734 = load float, float* %1733, align 4, !noalias !133
  %1735 = fadd float %1734, 3.000000e+00
  %1736 = fcmp olt float %1735, 0.000000e+00
  %1737 = select i1 %1736, float 0.000000e+00, float %1735
  %1738 = fcmp ogt float %1737, 6.000000e+00
  %1739 = select i1 %1738, float 6.000000e+00, float %1737
  %1740 = fmul float %1734, %1739
  %1741 = fdiv float %1740, 6.000000e+00
  %1742 = getelementptr float, float* %277, i64 %1732
  store float %1741, float* %1742, align 4, !noalias !133
  %1743 = add nuw nsw i64 %1693, 1
  %exitcond521.not.i = icmp eq i64 %1743, 14
  br i1 %exitcond521.not.i, label %exit245.i, label %cond246.preheader.i

exit245.i:                                        ; preds = %cond246.preheader.i
  %1744 = add nuw nsw i64 %1675, 1
  %exitcond522.not.i = icmp eq i64 %1744, 200
  br i1 %exitcond522.not.i, label %exit242.i, label %cond243.preheader.i

cond258.preheader.i:                              ; preds = %exit260.i, %exit242.i
  %1745 = phi i64 [ 0, %exit242.i ], [ %1814, %exit260.i ]
  %1746 = mul nuw nsw i64 %1745, 196
  %1747 = add nuw nsw i64 %1746, 8
  %1748 = add nuw nsw i64 %1746, 12
  br label %cond261.preheader.i

exit257.i:                                        ; preds = %exit260.i
  %1749 = alloca [3 x i8*], align 8
  %1750 = alloca [3 x i64], align 16
  %1751 = alloca [8 x i64], align 8
  %1752 = alloca [3 x i8], align 1
  %.sub173.i = getelementptr inbounds [3 x i8], [3 x i8]* %1752, i64 0, i64 0
  %.sub172.i = getelementptr inbounds [8 x i64], [8 x i64]* %1751, i64 0, i64 0
  %.sub171.i = getelementptr inbounds [3 x i64], [3 x i64]* %1750, i64 0, i64 0
  %.sub170.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %1749, i64 0, i64 0
  store i8* %malloccall90.i, i8** %.sub170.i, align 8, !noalias !133
  store i8 6, i8* %.sub173.i, align 1, !noalias !133
  %1753 = bitcast [8 x i64]* %1751 to <4 x i64>*
  store <4 x i64> <i64 1, i64 80, i64 14, i64 14>, <4 x i64>* %1753, align 8, !noalias !133
  %1754 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1749, i64 0, i64 1
  store i8* %malloccall24.i, i8** %1754, align 8, !noalias !133
  %1755 = getelementptr inbounds [3 x i8], [3 x i8]* %1752, i64 0, i64 1
  store i8 6, i8* %1755, align 1, !noalias !133
  %1756 = bitcast [3 x i64]* %1750 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %1756, align 16, !noalias !133
  %1757 = getelementptr inbounds [8 x i64], [8 x i64]* %1751, i64 0, i64 4
  %1758 = bitcast i64* %1757 to <4 x i64>*
  store <4 x i64> <i64 1, i64 200, i64 14, i64 14>, <4 x i64>* %1758, align 8, !noalias !133
  %1759 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1749, i64 0, i64 2
  %1760 = bitcast i8** %1759 to float**
  store float* %110, float** %1760, align 8, !noalias !133
  %1761 = getelementptr inbounds [3 x i8], [3 x i8]* %1752, i64 0, i64 2
  store i8 6, i8* %1761, align 1, !noalias !133
  %1762 = getelementptr inbounds [3 x i64], [3 x i64]* %1750, i64 0, i64 2
  store i64 0, i64* %1762, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub170.i, i64* nonnull %.sub171.i, i64* nonnull %.sub172.i, i8* nonnull %.sub173.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond273.preheader.i

cond261.preheader.i:                              ; preds = %cond261.preheader.i, %cond258.preheader.i
  %1763 = phi i64 [ 0, %cond258.preheader.i ], [ %1813, %cond261.preheader.i ]
  %1764 = mul nuw nsw i64 %1763, 14
  %1765 = add nuw nsw i64 %1764, %1746
  %1766 = getelementptr float, float* %257, i64 %1765
  %1767 = getelementptr float, float* %264, i64 %1765
  %1768 = bitcast float* %1766 to <8 x float>*
  %1769 = load <8 x float>, <8 x float>* %1768, align 4, !noalias !133
  %1770 = fadd <8 x float> %1769, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %1771 = fcmp olt <8 x float> %1770, zeroinitializer
  %1772 = select <8 x i1> %1771, <8 x float> zeroinitializer, <8 x float> %1770
  %1773 = fcmp ogt <8 x float> %1772, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1774 = select <8 x i1> %1773, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %1772
  %1775 = fmul <8 x float> %1769, %1774
  %1776 = fdiv <8 x float> %1775, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1777 = bitcast float* %1767 to <8 x float>*
  store <8 x float> %1776, <8 x float>* %1777, align 4, !noalias !133
  %1778 = add nuw nsw i64 %1747, %1764
  %1779 = getelementptr float, float* %257, i64 %1778
  %1780 = getelementptr float, float* %264, i64 %1778
  %1781 = bitcast float* %1779 to <4 x float>*
  %1782 = load <4 x float>, <4 x float>* %1781, align 4, !noalias !133
  %1783 = fadd <4 x float> %1782, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %1784 = fcmp olt <4 x float> %1783, zeroinitializer
  %1785 = select <4 x i1> %1784, <4 x float> zeroinitializer, <4 x float> %1783
  %1786 = fcmp ogt <4 x float> %1785, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1787 = select <4 x i1> %1786, <4 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <4 x float> %1785
  %1788 = fmul <4 x float> %1782, %1787
  %1789 = fdiv <4 x float> %1788, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1790 = bitcast float* %1780 to <4 x float>*
  store <4 x float> %1789, <4 x float>* %1790, align 4, !noalias !133
  %1791 = add nuw nsw i64 %1748, %1764
  %1792 = getelementptr float, float* %257, i64 %1791
  %1793 = load float, float* %1792, align 4, !noalias !133
  %1794 = fadd float %1793, 3.000000e+00
  %1795 = fcmp olt float %1794, 0.000000e+00
  %1796 = select i1 %1795, float 0.000000e+00, float %1794
  %1797 = fcmp ogt float %1796, 6.000000e+00
  %1798 = select i1 %1797, float 6.000000e+00, float %1796
  %1799 = fmul float %1793, %1798
  %1800 = fdiv float %1799, 6.000000e+00
  %1801 = getelementptr float, float* %264, i64 %1791
  store float %1800, float* %1801, align 4, !noalias !133
  %1802 = or i64 %1791, 1
  %1803 = getelementptr float, float* %257, i64 %1802
  %1804 = load float, float* %1803, align 4, !noalias !133
  %1805 = fadd float %1804, 3.000000e+00
  %1806 = fcmp olt float %1805, 0.000000e+00
  %1807 = select i1 %1806, float 0.000000e+00, float %1805
  %1808 = fcmp ogt float %1807, 6.000000e+00
  %1809 = select i1 %1808, float 6.000000e+00, float %1807
  %1810 = fmul float %1804, %1809
  %1811 = fdiv float %1810, 6.000000e+00
  %1812 = getelementptr float, float* %264, i64 %1802
  store float %1811, float* %1812, align 4, !noalias !133
  %1813 = add nuw nsw i64 %1763, 1
  %exitcond517.not.i = icmp eq i64 %1813, 14
  br i1 %exitcond517.not.i, label %exit260.i, label %cond261.preheader.i

exit260.i:                                        ; preds = %cond261.preheader.i
  %1814 = add nuw nsw i64 %1745, 1
  %exitcond518.not.i = icmp eq i64 %1814, 200
  br i1 %exitcond518.not.i, label %exit257.i, label %cond258.preheader.i

cond273.preheader.i:                              ; preds = %exit275.i, %exit257.i
  %1815 = phi i64 [ 0, %exit257.i ], [ %1906, %exit275.i ]
  %1816 = mul nuw nsw i64 %1815, 196
  %1817 = add nuw nsw i64 %1816, 8
  %1818 = add nuw nsw i64 %1816, 12
  br label %cond276.preheader.i

exit272.i:                                        ; preds = %exit275.i
  %1819 = alloca [3 x i8*], align 8
  %1820 = alloca [3 x i64], align 16
  %1821 = alloca [8 x i64], align 8
  %1822 = alloca [3 x i8], align 1
  %.sub178.i = getelementptr inbounds [3 x i8], [3 x i8]* %1822, i64 0, i64 0
  %.sub177.i = getelementptr inbounds [8 x i64], [8 x i64]* %1821, i64 0, i64 0
  %.sub176.i = getelementptr inbounds [3 x i64], [3 x i64]* %1820, i64 0, i64 0
  %.sub175.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %1819, i64 0, i64 0
  store i8* %malloccall45.i, i8** %.sub175.i, align 8, !noalias !133
  store i8 6, i8* %.sub178.i, align 1, !noalias !133
  %1823 = bitcast [8 x i64]* %1821 to <4 x i64>*
  store <4 x i64> <i64 1, i64 184, i64 14, i64 14>, <4 x i64>* %1823, align 8, !noalias !133
  %1824 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1819, i64 0, i64 1
  store i8* %malloccall60.i, i8** %1824, align 8, !noalias !133
  %1825 = getelementptr inbounds [3 x i8], [3 x i8]* %1822, i64 0, i64 1
  store i8 6, i8* %1825, align 1, !noalias !133
  %1826 = bitcast [3 x i64]* %1820 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %1826, align 16, !noalias !133
  %1827 = getelementptr inbounds [8 x i64], [8 x i64]* %1821, i64 0, i64 4
  %1828 = bitcast i64* %1827 to <4 x i64>*
  store <4 x i64> <i64 1, i64 80, i64 14, i64 14>, <4 x i64>* %1828, align 8, !noalias !133
  %1829 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1819, i64 0, i64 2
  %1830 = bitcast i8** %1829 to float**
  store float* %113, float** %1830, align 8, !noalias !133
  %1831 = getelementptr inbounds [3 x i8], [3 x i8]* %1822, i64 0, i64 2
  store i8 6, i8* %1831, align 1, !noalias !133
  %1832 = getelementptr inbounds [3 x i64], [3 x i64]* %1820, i64 0, i64 2
  store i64 0, i64* %1832, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub175.i, i64* nonnull %.sub176.i, i64* nonnull %.sub177.i, i8* nonnull %.sub178.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond288.preheader.i

cond276.preheader.i:                              ; preds = %cond276.preheader.i, %cond273.preheader.i
  %1833 = phi i64 [ 0, %cond273.preheader.i ], [ %1905, %cond276.preheader.i ]
  %1834 = mul nuw nsw i64 %1833, 14
  %1835 = add nuw nsw i64 %1834, %1816
  %1836 = getelementptr float, float* %315, i64 %1835
  %1837 = getelementptr float, float* %274, i64 %1835
  %1838 = getelementptr float, float* %292, i64 %1835
  %1839 = bitcast float* %1836 to <8 x float>*
  %1840 = load <8 x float>, <8 x float>* %1839, align 4, !noalias !133
  %1841 = bitcast float* %1837 to <8 x float>*
  %1842 = load <8 x float>, <8 x float>* %1841, align 4, !noalias !133
  %1843 = fadd <8 x float> %1840, %1842
  %1844 = bitcast float* %1838 to <8 x float>*
  store <8 x float> %1843, <8 x float>* %1844, align 4, !noalias !133
  %1845 = add nuw nsw i64 %1817, %1834
  %1846 = getelementptr float, float* %315, i64 %1845
  %1847 = getelementptr float, float* %274, i64 %1845
  %1848 = getelementptr float, float* %292, i64 %1845
  %1849 = bitcast float* %1846 to <4 x float>*
  %1850 = load <4 x float>, <4 x float>* %1849, align 4, !noalias !133
  %1851 = bitcast float* %1847 to <4 x float>*
  %1852 = load <4 x float>, <4 x float>* %1851, align 4, !noalias !133
  %1853 = fadd <4 x float> %1850, %1852
  %1854 = bitcast float* %1848 to <4 x float>*
  store <4 x float> %1853, <4 x float>* %1854, align 4, !noalias !133
  %1855 = add nuw nsw i64 %1818, %1834
  %1856 = getelementptr float, float* %315, i64 %1855
  %1857 = load float, float* %1856, align 4, !noalias !133
  %1858 = getelementptr float, float* %274, i64 %1855
  %1859 = load float, float* %1858, align 4, !noalias !133
  %1860 = fadd float %1857, %1859
  %1861 = getelementptr float, float* %292, i64 %1855
  store float %1860, float* %1861, align 4, !noalias !133
  %1862 = or i64 %1855, 1
  %1863 = getelementptr float, float* %315, i64 %1862
  %1864 = load float, float* %1863, align 4, !noalias !133
  %1865 = getelementptr float, float* %274, i64 %1862
  %1866 = load float, float* %1865, align 4, !noalias !133
  %1867 = fadd float %1864, %1866
  %1868 = getelementptr float, float* %292, i64 %1862
  store float %1867, float* %1868, align 4, !noalias !133
  %1869 = or i64 %1833, 1
  %1870 = mul nuw nsw i64 %1869, 14
  %1871 = add nuw nsw i64 %1870, %1816
  %1872 = getelementptr float, float* %315, i64 %1871
  %1873 = getelementptr float, float* %274, i64 %1871
  %1874 = getelementptr float, float* %292, i64 %1871
  %1875 = bitcast float* %1872 to <8 x float>*
  %1876 = load <8 x float>, <8 x float>* %1875, align 4, !noalias !133
  %1877 = bitcast float* %1873 to <8 x float>*
  %1878 = load <8 x float>, <8 x float>* %1877, align 4, !noalias !133
  %1879 = fadd <8 x float> %1876, %1878
  %1880 = bitcast float* %1874 to <8 x float>*
  store <8 x float> %1879, <8 x float>* %1880, align 4, !noalias !133
  %1881 = add nuw nsw i64 %1817, %1870
  %1882 = getelementptr float, float* %315, i64 %1881
  %1883 = getelementptr float, float* %274, i64 %1881
  %1884 = getelementptr float, float* %292, i64 %1881
  %1885 = bitcast float* %1882 to <4 x float>*
  %1886 = load <4 x float>, <4 x float>* %1885, align 4, !noalias !133
  %1887 = bitcast float* %1883 to <4 x float>*
  %1888 = load <4 x float>, <4 x float>* %1887, align 4, !noalias !133
  %1889 = fadd <4 x float> %1886, %1888
  %1890 = bitcast float* %1884 to <4 x float>*
  store <4 x float> %1889, <4 x float>* %1890, align 4, !noalias !133
  %1891 = add nuw nsw i64 %1818, %1870
  %1892 = getelementptr float, float* %315, i64 %1891
  %1893 = load float, float* %1892, align 4, !noalias !133
  %1894 = getelementptr float, float* %274, i64 %1891
  %1895 = load float, float* %1894, align 4, !noalias !133
  %1896 = fadd float %1893, %1895
  %1897 = getelementptr float, float* %292, i64 %1891
  store float %1896, float* %1897, align 4, !noalias !133
  %1898 = or i64 %1891, 1
  %1899 = getelementptr float, float* %315, i64 %1898
  %1900 = load float, float* %1899, align 4, !noalias !133
  %1901 = getelementptr float, float* %274, i64 %1898
  %1902 = load float, float* %1901, align 4, !noalias !133
  %1903 = fadd float %1900, %1902
  %1904 = getelementptr float, float* %292, i64 %1898
  store float %1903, float* %1904, align 4, !noalias !133
  %1905 = add nuw nsw i64 %1833, 2
  %exitcond513.not.1.i = icmp eq i64 %1905, 14
  br i1 %exitcond513.not.1.i, label %exit275.i, label %cond276.preheader.i

exit275.i:                                        ; preds = %cond276.preheader.i
  %1906 = add nuw nsw i64 %1815, 1
  %exitcond514.not.i = icmp eq i64 %1906, 80
  br i1 %exitcond514.not.i, label %exit272.i, label %cond273.preheader.i

cond288.preheader.i:                              ; preds = %exit290.i, %exit272.i
  %1907 = phi i64 [ 0, %exit272.i ], [ %1976, %exit290.i ]
  %1908 = mul nuw nsw i64 %1907, 196
  %1909 = add nuw nsw i64 %1908, 8
  %1910 = add nuw nsw i64 %1908, 12
  br label %cond291.preheader.i

exit287.i:                                        ; preds = %exit290.i
  %1911 = alloca [3 x i8*], align 8
  %1912 = alloca [3 x i64], align 16
  %1913 = alloca [8 x i64], align 8
  %1914 = alloca [3 x i8], align 1
  %.sub183.i = getelementptr inbounds [3 x i8], [3 x i8]* %1914, i64 0, i64 0
  %.sub182.i = getelementptr inbounds [8 x i64], [8 x i64]* %1913, i64 0, i64 0
  %.sub181.i = getelementptr inbounds [3 x i64], [3 x i64]* %1912, i64 0, i64 0
  %.sub180.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %1911, i64 0, i64 0
  store i8* %malloccall30.i, i8** %.sub180.i, align 8, !noalias !133
  store i8 6, i8* %.sub183.i, align 1, !noalias !133
  %1915 = bitcast [8 x i64]* %1913 to <4 x i64>*
  store <4 x i64> <i64 1, i64 184, i64 14, i64 14>, <4 x i64>* %1915, align 8, !noalias !133
  %1916 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1911, i64 0, i64 1
  store i8* %malloccall31.i, i8** %1916, align 8, !noalias !133
  %1917 = getelementptr inbounds [3 x i8], [3 x i8]* %1914, i64 0, i64 1
  store i8 6, i8* %1917, align 1, !noalias !133
  %1918 = bitcast [3 x i64]* %1912 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %1918, align 16, !noalias !133
  %1919 = getelementptr inbounds [8 x i64], [8 x i64]* %1913, i64 0, i64 4
  %1920 = bitcast i64* %1919 to <4 x i64>*
  store <4 x i64> <i64 1, i64 184, i64 14, i64 14>, <4 x i64>* %1920, align 8, !noalias !133
  %1921 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1911, i64 0, i64 2
  %1922 = bitcast i8** %1921 to float**
  store float* %116, float** %1922, align 8, !noalias !133
  %1923 = getelementptr inbounds [3 x i8], [3 x i8]* %1914, i64 0, i64 2
  store i8 6, i8* %1923, align 1, !noalias !133
  %1924 = getelementptr inbounds [3 x i64], [3 x i64]* %1912, i64 0, i64 2
  store i64 0, i64* %1924, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub180.i, i64* nonnull %.sub181.i, i64* nonnull %.sub182.i, i8* nonnull %.sub183.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond303.preheader.i

cond291.preheader.i:                              ; preds = %cond291.preheader.i, %cond288.preheader.i
  %1925 = phi i64 [ 0, %cond288.preheader.i ], [ %1975, %cond291.preheader.i ]
  %1926 = mul nuw nsw i64 %1925, 14
  %1927 = add nuw nsw i64 %1926, %1908
  %1928 = getelementptr float, float* %281, i64 %1927
  %1929 = getelementptr float, float* %271, i64 %1927
  %1930 = bitcast float* %1928 to <8 x float>*
  %1931 = load <8 x float>, <8 x float>* %1930, align 4, !noalias !133
  %1932 = fadd <8 x float> %1931, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %1933 = fcmp olt <8 x float> %1932, zeroinitializer
  %1934 = select <8 x i1> %1933, <8 x float> zeroinitializer, <8 x float> %1932
  %1935 = fcmp ogt <8 x float> %1934, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1936 = select <8 x i1> %1935, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %1934
  %1937 = fmul <8 x float> %1931, %1936
  %1938 = fdiv <8 x float> %1937, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1939 = bitcast float* %1929 to <8 x float>*
  store <8 x float> %1938, <8 x float>* %1939, align 4, !noalias !133
  %1940 = add nuw nsw i64 %1909, %1926
  %1941 = getelementptr float, float* %281, i64 %1940
  %1942 = getelementptr float, float* %271, i64 %1940
  %1943 = bitcast float* %1941 to <4 x float>*
  %1944 = load <4 x float>, <4 x float>* %1943, align 4, !noalias !133
  %1945 = fadd <4 x float> %1944, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %1946 = fcmp olt <4 x float> %1945, zeroinitializer
  %1947 = select <4 x i1> %1946, <4 x float> zeroinitializer, <4 x float> %1945
  %1948 = fcmp ogt <4 x float> %1947, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1949 = select <4 x i1> %1948, <4 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <4 x float> %1947
  %1950 = fmul <4 x float> %1944, %1949
  %1951 = fdiv <4 x float> %1950, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %1952 = bitcast float* %1942 to <4 x float>*
  store <4 x float> %1951, <4 x float>* %1952, align 4, !noalias !133
  %1953 = add nuw nsw i64 %1910, %1926
  %1954 = getelementptr float, float* %281, i64 %1953
  %1955 = load float, float* %1954, align 4, !noalias !133
  %1956 = fadd float %1955, 3.000000e+00
  %1957 = fcmp olt float %1956, 0.000000e+00
  %1958 = select i1 %1957, float 0.000000e+00, float %1956
  %1959 = fcmp ogt float %1958, 6.000000e+00
  %1960 = select i1 %1959, float 6.000000e+00, float %1958
  %1961 = fmul float %1955, %1960
  %1962 = fdiv float %1961, 6.000000e+00
  %1963 = getelementptr float, float* %271, i64 %1953
  store float %1962, float* %1963, align 4, !noalias !133
  %1964 = or i64 %1953, 1
  %1965 = getelementptr float, float* %281, i64 %1964
  %1966 = load float, float* %1965, align 4, !noalias !133
  %1967 = fadd float %1966, 3.000000e+00
  %1968 = fcmp olt float %1967, 0.000000e+00
  %1969 = select i1 %1968, float 0.000000e+00, float %1967
  %1970 = fcmp ogt float %1969, 6.000000e+00
  %1971 = select i1 %1970, float 6.000000e+00, float %1969
  %1972 = fmul float %1966, %1971
  %1973 = fdiv float %1972, 6.000000e+00
  %1974 = getelementptr float, float* %271, i64 %1964
  store float %1973, float* %1974, align 4, !noalias !133
  %1975 = add nuw nsw i64 %1925, 1
  %exitcond509.not.i = icmp eq i64 %1975, 14
  br i1 %exitcond509.not.i, label %exit290.i, label %cond291.preheader.i

exit290.i:                                        ; preds = %cond291.preheader.i
  %1976 = add nuw nsw i64 %1907, 1
  %exitcond510.not.i = icmp eq i64 %1976, 184
  br i1 %exitcond510.not.i, label %exit287.i, label %cond288.preheader.i

cond303.preheader.i:                              ; preds = %exit305.i, %exit287.i
  %1977 = phi i64 [ 0, %exit287.i ], [ %2046, %exit305.i ]
  %1978 = mul nuw nsw i64 %1977, 196
  %1979 = add nuw nsw i64 %1978, 8
  %1980 = add nuw nsw i64 %1978, 12
  br label %cond306.preheader.i

exit302.i:                                        ; preds = %exit305.i
  %1981 = alloca [3 x i8*], align 8
  %1982 = alloca [3 x i64], align 16
  %1983 = alloca [8 x i64], align 8
  %1984 = alloca [3 x i8], align 1
  %.sub188.i = getelementptr inbounds [3 x i8], [3 x i8]* %1984, i64 0, i64 0
  %.sub187.i = getelementptr inbounds [8 x i64], [8 x i64]* %1983, i64 0, i64 0
  %.sub186.i = getelementptr inbounds [3 x i64], [3 x i64]* %1982, i64 0, i64 0
  %.sub185.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %1981, i64 0, i64 0
  store i8* %malloccall27.i, i8** %.sub185.i, align 8, !noalias !133
  store i8 6, i8* %.sub188.i, align 1, !noalias !133
  %1985 = bitcast [8 x i64]* %1983 to <4 x i64>*
  store <4 x i64> <i64 1, i64 80, i64 14, i64 14>, <4 x i64>* %1985, align 8, !noalias !133
  %1986 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1981, i64 0, i64 1
  store i8* %malloccall28.i, i8** %1986, align 8, !noalias !133
  %1987 = getelementptr inbounds [3 x i8], [3 x i8]* %1984, i64 0, i64 1
  store i8 6, i8* %1987, align 1, !noalias !133
  %1988 = bitcast [3 x i64]* %1982 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %1988, align 16, !noalias !133
  %1989 = getelementptr inbounds [8 x i64], [8 x i64]* %1983, i64 0, i64 4
  %1990 = bitcast i64* %1989 to <4 x i64>*
  store <4 x i64> <i64 1, i64 184, i64 14, i64 14>, <4 x i64>* %1990, align 8, !noalias !133
  %1991 = getelementptr inbounds [3 x i8*], [3 x i8*]* %1981, i64 0, i64 2
  %1992 = bitcast i8** %1991 to float**
  store float* %119, float** %1992, align 8, !noalias !133
  %1993 = getelementptr inbounds [3 x i8], [3 x i8]* %1984, i64 0, i64 2
  store i8 6, i8* %1993, align 1, !noalias !133
  %1994 = getelementptr inbounds [3 x i64], [3 x i64]* %1982, i64 0, i64 2
  store i64 0, i64* %1994, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub185.i, i64* nonnull %.sub186.i, i64* nonnull %.sub187.i, i8* nonnull %.sub188.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond318.preheader.i

cond306.preheader.i:                              ; preds = %cond306.preheader.i, %cond303.preheader.i
  %1995 = phi i64 [ 0, %cond303.preheader.i ], [ %2045, %cond306.preheader.i ]
  %1996 = mul nuw nsw i64 %1995, 14
  %1997 = add nuw nsw i64 %1996, %1978
  %1998 = getelementptr float, float* %270, i64 %1997
  %1999 = getelementptr float, float* %268, i64 %1997
  %2000 = bitcast float* %1998 to <8 x float>*
  %2001 = load <8 x float>, <8 x float>* %2000, align 4, !noalias !133
  %2002 = fadd <8 x float> %2001, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %2003 = fcmp olt <8 x float> %2002, zeroinitializer
  %2004 = select <8 x i1> %2003, <8 x float> zeroinitializer, <8 x float> %2002
  %2005 = fcmp ogt <8 x float> %2004, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2006 = select <8 x i1> %2005, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %2004
  %2007 = fmul <8 x float> %2001, %2006
  %2008 = fdiv <8 x float> %2007, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2009 = bitcast float* %1999 to <8 x float>*
  store <8 x float> %2008, <8 x float>* %2009, align 4, !noalias !133
  %2010 = add nuw nsw i64 %1979, %1996
  %2011 = getelementptr float, float* %270, i64 %2010
  %2012 = getelementptr float, float* %268, i64 %2010
  %2013 = bitcast float* %2011 to <4 x float>*
  %2014 = load <4 x float>, <4 x float>* %2013, align 4, !noalias !133
  %2015 = fadd <4 x float> %2014, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %2016 = fcmp olt <4 x float> %2015, zeroinitializer
  %2017 = select <4 x i1> %2016, <4 x float> zeroinitializer, <4 x float> %2015
  %2018 = fcmp ogt <4 x float> %2017, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2019 = select <4 x i1> %2018, <4 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <4 x float> %2017
  %2020 = fmul <4 x float> %2014, %2019
  %2021 = fdiv <4 x float> %2020, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2022 = bitcast float* %2012 to <4 x float>*
  store <4 x float> %2021, <4 x float>* %2022, align 4, !noalias !133
  %2023 = add nuw nsw i64 %1980, %1996
  %2024 = getelementptr float, float* %270, i64 %2023
  %2025 = load float, float* %2024, align 4, !noalias !133
  %2026 = fadd float %2025, 3.000000e+00
  %2027 = fcmp olt float %2026, 0.000000e+00
  %2028 = select i1 %2027, float 0.000000e+00, float %2026
  %2029 = fcmp ogt float %2028, 6.000000e+00
  %2030 = select i1 %2029, float 6.000000e+00, float %2028
  %2031 = fmul float %2025, %2030
  %2032 = fdiv float %2031, 6.000000e+00
  %2033 = getelementptr float, float* %268, i64 %2023
  store float %2032, float* %2033, align 4, !noalias !133
  %2034 = or i64 %2023, 1
  %2035 = getelementptr float, float* %270, i64 %2034
  %2036 = load float, float* %2035, align 4, !noalias !133
  %2037 = fadd float %2036, 3.000000e+00
  %2038 = fcmp olt float %2037, 0.000000e+00
  %2039 = select i1 %2038, float 0.000000e+00, float %2037
  %2040 = fcmp ogt float %2039, 6.000000e+00
  %2041 = select i1 %2040, float 6.000000e+00, float %2039
  %2042 = fmul float %2036, %2041
  %2043 = fdiv float %2042, 6.000000e+00
  %2044 = getelementptr float, float* %268, i64 %2034
  store float %2043, float* %2044, align 4, !noalias !133
  %2045 = add nuw nsw i64 %1995, 1
  %exitcond505.not.i = icmp eq i64 %2045, 14
  br i1 %exitcond505.not.i, label %exit305.i, label %cond306.preheader.i

exit305.i:                                        ; preds = %cond306.preheader.i
  %2046 = add nuw nsw i64 %1977, 1
  %exitcond506.not.i = icmp eq i64 %2046, 184
  br i1 %exitcond506.not.i, label %exit302.i, label %cond303.preheader.i

cond318.preheader.i:                              ; preds = %exit320.i, %exit302.i
  %2047 = phi i64 [ 0, %exit302.i ], [ %2138, %exit320.i ]
  %2048 = mul nuw nsw i64 %2047, 196
  %2049 = add nuw nsw i64 %2048, 8
  %2050 = add nuw nsw i64 %2048, 12
  br label %cond321.preheader.i

exit317.i:                                        ; preds = %exit320.i
  %2051 = alloca [3 x i8*], align 8
  %2052 = alloca [3 x i64], align 16
  %2053 = alloca [8 x i64], align 8
  %2054 = alloca [3 x i8], align 1
  %.sub193.i = getelementptr inbounds [3 x i8], [3 x i8]* %2054, i64 0, i64 0
  %.sub192.i = getelementptr inbounds [8 x i64], [8 x i64]* %2053, i64 0, i64 0
  %.sub191.i = getelementptr inbounds [3 x i64], [3 x i64]* %2052, i64 0, i64 0
  %.sub190.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %2051, i64 0, i64 0
  store i8* %malloccall26.i, i8** %.sub190.i, align 8, !noalias !133
  store i8 6, i8* %.sub193.i, align 1, !noalias !133
  %2055 = bitcast [8 x i64]* %2053 to <4 x i64>*
  store <4 x i64> <i64 1, i64 184, i64 14, i64 14>, <4 x i64>* %2055, align 8, !noalias !133
  %2056 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2051, i64 0, i64 1
  store i8* %malloccall58.i, i8** %2056, align 8, !noalias !133
  %2057 = getelementptr inbounds [3 x i8], [3 x i8]* %2054, i64 0, i64 1
  store i8 6, i8* %2057, align 1, !noalias !133
  %2058 = bitcast [3 x i64]* %2052 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %2058, align 16, !noalias !133
  %2059 = getelementptr inbounds [8 x i64], [8 x i64]* %2053, i64 0, i64 4
  %2060 = bitcast i64* %2059 to <4 x i64>*
  store <4 x i64> <i64 1, i64 80, i64 14, i64 14>, <4 x i64>* %2060, align 8, !noalias !133
  %2061 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2051, i64 0, i64 2
  %2062 = bitcast i8** %2061 to float**
  store float* %122, float** %2062, align 8, !noalias !133
  %2063 = getelementptr inbounds [3 x i8], [3 x i8]* %2054, i64 0, i64 2
  store i8 6, i8* %2063, align 1, !noalias !133
  %2064 = getelementptr inbounds [3 x i64], [3 x i64]* %2052, i64 0, i64 2
  store i64 0, i64* %2064, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub190.i, i64* nonnull %.sub191.i, i64* nonnull %.sub192.i, i8* nonnull %.sub193.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond333.preheader.i

cond321.preheader.i:                              ; preds = %cond321.preheader.i, %cond318.preheader.i
  %2065 = phi i64 [ 0, %cond318.preheader.i ], [ %2137, %cond321.preheader.i ]
  %2066 = mul nuw nsw i64 %2065, 14
  %2067 = add nuw nsw i64 %2066, %2048
  %2068 = getelementptr float, float* %267, i64 %2067
  %2069 = getelementptr float, float* %292, i64 %2067
  %2070 = getelementptr float, float* %291, i64 %2067
  %2071 = bitcast float* %2068 to <8 x float>*
  %2072 = load <8 x float>, <8 x float>* %2071, align 4, !noalias !133
  %2073 = bitcast float* %2069 to <8 x float>*
  %2074 = load <8 x float>, <8 x float>* %2073, align 4, !noalias !133
  %2075 = fadd <8 x float> %2072, %2074
  %2076 = bitcast float* %2070 to <8 x float>*
  store <8 x float> %2075, <8 x float>* %2076, align 4, !noalias !133
  %2077 = add nuw nsw i64 %2049, %2066
  %2078 = getelementptr float, float* %267, i64 %2077
  %2079 = getelementptr float, float* %292, i64 %2077
  %2080 = getelementptr float, float* %291, i64 %2077
  %2081 = bitcast float* %2078 to <4 x float>*
  %2082 = load <4 x float>, <4 x float>* %2081, align 4, !noalias !133
  %2083 = bitcast float* %2079 to <4 x float>*
  %2084 = load <4 x float>, <4 x float>* %2083, align 4, !noalias !133
  %2085 = fadd <4 x float> %2082, %2084
  %2086 = bitcast float* %2080 to <4 x float>*
  store <4 x float> %2085, <4 x float>* %2086, align 4, !noalias !133
  %2087 = add nuw nsw i64 %2050, %2066
  %2088 = getelementptr float, float* %267, i64 %2087
  %2089 = load float, float* %2088, align 4, !noalias !133
  %2090 = getelementptr float, float* %292, i64 %2087
  %2091 = load float, float* %2090, align 4, !noalias !133
  %2092 = fadd float %2089, %2091
  %2093 = getelementptr float, float* %291, i64 %2087
  store float %2092, float* %2093, align 4, !noalias !133
  %2094 = or i64 %2087, 1
  %2095 = getelementptr float, float* %267, i64 %2094
  %2096 = load float, float* %2095, align 4, !noalias !133
  %2097 = getelementptr float, float* %292, i64 %2094
  %2098 = load float, float* %2097, align 4, !noalias !133
  %2099 = fadd float %2096, %2098
  %2100 = getelementptr float, float* %291, i64 %2094
  store float %2099, float* %2100, align 4, !noalias !133
  %2101 = or i64 %2065, 1
  %2102 = mul nuw nsw i64 %2101, 14
  %2103 = add nuw nsw i64 %2102, %2048
  %2104 = getelementptr float, float* %267, i64 %2103
  %2105 = getelementptr float, float* %292, i64 %2103
  %2106 = getelementptr float, float* %291, i64 %2103
  %2107 = bitcast float* %2104 to <8 x float>*
  %2108 = load <8 x float>, <8 x float>* %2107, align 4, !noalias !133
  %2109 = bitcast float* %2105 to <8 x float>*
  %2110 = load <8 x float>, <8 x float>* %2109, align 4, !noalias !133
  %2111 = fadd <8 x float> %2108, %2110
  %2112 = bitcast float* %2106 to <8 x float>*
  store <8 x float> %2111, <8 x float>* %2112, align 4, !noalias !133
  %2113 = add nuw nsw i64 %2049, %2102
  %2114 = getelementptr float, float* %267, i64 %2113
  %2115 = getelementptr float, float* %292, i64 %2113
  %2116 = getelementptr float, float* %291, i64 %2113
  %2117 = bitcast float* %2114 to <4 x float>*
  %2118 = load <4 x float>, <4 x float>* %2117, align 4, !noalias !133
  %2119 = bitcast float* %2115 to <4 x float>*
  %2120 = load <4 x float>, <4 x float>* %2119, align 4, !noalias !133
  %2121 = fadd <4 x float> %2118, %2120
  %2122 = bitcast float* %2116 to <4 x float>*
  store <4 x float> %2121, <4 x float>* %2122, align 4, !noalias !133
  %2123 = add nuw nsw i64 %2050, %2102
  %2124 = getelementptr float, float* %267, i64 %2123
  %2125 = load float, float* %2124, align 4, !noalias !133
  %2126 = getelementptr float, float* %292, i64 %2123
  %2127 = load float, float* %2126, align 4, !noalias !133
  %2128 = fadd float %2125, %2127
  %2129 = getelementptr float, float* %291, i64 %2123
  store float %2128, float* %2129, align 4, !noalias !133
  %2130 = or i64 %2123, 1
  %2131 = getelementptr float, float* %267, i64 %2130
  %2132 = load float, float* %2131, align 4, !noalias !133
  %2133 = getelementptr float, float* %292, i64 %2130
  %2134 = load float, float* %2133, align 4, !noalias !133
  %2135 = fadd float %2132, %2134
  %2136 = getelementptr float, float* %291, i64 %2130
  store float %2135, float* %2136, align 4, !noalias !133
  %2137 = add nuw nsw i64 %2065, 2
  %exitcond501.not.1.i = icmp eq i64 %2137, 14
  br i1 %exitcond501.not.1.i, label %exit320.i, label %cond321.preheader.i

exit320.i:                                        ; preds = %cond321.preheader.i
  %2138 = add nuw nsw i64 %2047, 1
  %exitcond502.not.i = icmp eq i64 %2138, 80
  br i1 %exitcond502.not.i, label %exit317.i, label %cond318.preheader.i

cond333.preheader.i:                              ; preds = %exit335.i, %exit317.i
  %2139 = phi i64 [ 0, %exit317.i ], [ %2208, %exit335.i ]
  %2140 = mul nuw nsw i64 %2139, 196
  %2141 = add nuw nsw i64 %2140, 8
  %2142 = add nuw nsw i64 %2140, 12
  br label %cond336.preheader.i

exit332.i:                                        ; preds = %exit335.i
  %2143 = alloca [3 x i8*], align 8
  %2144 = alloca [3 x i64], align 16
  %2145 = alloca [8 x i64], align 8
  %2146 = alloca [3 x i8], align 1
  %.sub198.i = getelementptr inbounds [3 x i8], [3 x i8]* %2146, i64 0, i64 0
  %.sub197.i = getelementptr inbounds [8 x i64], [8 x i64]* %2145, i64 0, i64 0
  %.sub196.i = getelementptr inbounds [3 x i64], [3 x i64]* %2144, i64 0, i64 0
  %.sub195.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %2143, i64 0, i64 0
  store i8* %malloccall49.i, i8** %.sub195.i, align 8, !noalias !133
  store i8 6, i8* %.sub198.i, align 1, !noalias !133
  %2147 = bitcast [8 x i64]* %2145 to <4 x i64>*
  store <4 x i64> <i64 1, i64 184, i64 14, i64 14>, <4 x i64>* %2147, align 8, !noalias !133
  %2148 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2143, i64 0, i64 1
  store i8* %malloccall23.i, i8** %2148, align 8, !noalias !133
  %2149 = getelementptr inbounds [3 x i8], [3 x i8]* %2146, i64 0, i64 1
  store i8 6, i8* %2149, align 1, !noalias !133
  %2150 = bitcast [3 x i64]* %2144 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %2150, align 16, !noalias !133
  %2151 = getelementptr inbounds [8 x i64], [8 x i64]* %2145, i64 0, i64 4
  %2152 = bitcast i64* %2151 to <4 x i64>*
  store <4 x i64> <i64 1, i64 184, i64 14, i64 14>, <4 x i64>* %2152, align 8, !noalias !133
  %2153 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2143, i64 0, i64 2
  %2154 = bitcast i8** %2153 to float**
  store float* %125, float** %2154, align 8, !noalias !133
  %2155 = getelementptr inbounds [3 x i8], [3 x i8]* %2146, i64 0, i64 2
  store i8 6, i8* %2155, align 1, !noalias !133
  %2156 = getelementptr inbounds [3 x i64], [3 x i64]* %2144, i64 0, i64 2
  store i64 0, i64* %2156, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub195.i, i64* nonnull %.sub196.i, i64* nonnull %.sub197.i, i8* nonnull %.sub198.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond348.preheader.i

cond336.preheader.i:                              ; preds = %cond336.preheader.i, %cond333.preheader.i
  %2157 = phi i64 [ 0, %cond333.preheader.i ], [ %2207, %cond336.preheader.i ]
  %2158 = mul nuw nsw i64 %2157, 14
  %2159 = add nuw nsw i64 %2158, %2140
  %2160 = getelementptr float, float* %266, i64 %2159
  %2161 = getelementptr float, float* %263, i64 %2159
  %2162 = bitcast float* %2160 to <8 x float>*
  %2163 = load <8 x float>, <8 x float>* %2162, align 4, !noalias !133
  %2164 = fadd <8 x float> %2163, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %2165 = fcmp olt <8 x float> %2164, zeroinitializer
  %2166 = select <8 x i1> %2165, <8 x float> zeroinitializer, <8 x float> %2164
  %2167 = fcmp ogt <8 x float> %2166, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2168 = select <8 x i1> %2167, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %2166
  %2169 = fmul <8 x float> %2163, %2168
  %2170 = fdiv <8 x float> %2169, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2171 = bitcast float* %2161 to <8 x float>*
  store <8 x float> %2170, <8 x float>* %2171, align 4, !noalias !133
  %2172 = add nuw nsw i64 %2141, %2158
  %2173 = getelementptr float, float* %266, i64 %2172
  %2174 = getelementptr float, float* %263, i64 %2172
  %2175 = bitcast float* %2173 to <4 x float>*
  %2176 = load <4 x float>, <4 x float>* %2175, align 4, !noalias !133
  %2177 = fadd <4 x float> %2176, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %2178 = fcmp olt <4 x float> %2177, zeroinitializer
  %2179 = select <4 x i1> %2178, <4 x float> zeroinitializer, <4 x float> %2177
  %2180 = fcmp ogt <4 x float> %2179, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2181 = select <4 x i1> %2180, <4 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <4 x float> %2179
  %2182 = fmul <4 x float> %2176, %2181
  %2183 = fdiv <4 x float> %2182, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2184 = bitcast float* %2174 to <4 x float>*
  store <4 x float> %2183, <4 x float>* %2184, align 4, !noalias !133
  %2185 = add nuw nsw i64 %2142, %2158
  %2186 = getelementptr float, float* %266, i64 %2185
  %2187 = load float, float* %2186, align 4, !noalias !133
  %2188 = fadd float %2187, 3.000000e+00
  %2189 = fcmp olt float %2188, 0.000000e+00
  %2190 = select i1 %2189, float 0.000000e+00, float %2188
  %2191 = fcmp ogt float %2190, 6.000000e+00
  %2192 = select i1 %2191, float 6.000000e+00, float %2190
  %2193 = fmul float %2187, %2192
  %2194 = fdiv float %2193, 6.000000e+00
  %2195 = getelementptr float, float* %263, i64 %2185
  store float %2194, float* %2195, align 4, !noalias !133
  %2196 = or i64 %2185, 1
  %2197 = getelementptr float, float* %266, i64 %2196
  %2198 = load float, float* %2197, align 4, !noalias !133
  %2199 = fadd float %2198, 3.000000e+00
  %2200 = fcmp olt float %2199, 0.000000e+00
  %2201 = select i1 %2200, float 0.000000e+00, float %2199
  %2202 = fcmp ogt float %2201, 6.000000e+00
  %2203 = select i1 %2202, float 6.000000e+00, float %2201
  %2204 = fmul float %2198, %2203
  %2205 = fdiv float %2204, 6.000000e+00
  %2206 = getelementptr float, float* %263, i64 %2196
  store float %2205, float* %2206, align 4, !noalias !133
  %2207 = add nuw nsw i64 %2157, 1
  %exitcond497.not.i = icmp eq i64 %2207, 14
  br i1 %exitcond497.not.i, label %exit335.i, label %cond336.preheader.i

exit335.i:                                        ; preds = %cond336.preheader.i
  %2208 = add nuw nsw i64 %2139, 1
  %exitcond498.not.i = icmp eq i64 %2208, 184
  br i1 %exitcond498.not.i, label %exit332.i, label %cond333.preheader.i

cond348.preheader.i:                              ; preds = %exit350.i, %exit332.i
  %2209 = phi i64 [ 0, %exit332.i ], [ %2278, %exit350.i ]
  %2210 = mul nuw nsw i64 %2209, 196
  %2211 = add nuw nsw i64 %2210, 8
  %2212 = add nuw nsw i64 %2210, 12
  br label %cond351.preheader.i

exit347.i:                                        ; preds = %exit350.i
  %2213 = alloca [3 x i8*], align 8
  %2214 = alloca [3 x i64], align 16
  %2215 = alloca [8 x i64], align 8
  %2216 = alloca [3 x i8], align 1
  %.sub203.i = getelementptr inbounds [3 x i8], [3 x i8]* %2216, i64 0, i64 0
  %.sub202.i = getelementptr inbounds [8 x i64], [8 x i64]* %2215, i64 0, i64 0
  %.sub201.i = getelementptr inbounds [3 x i64], [3 x i64]* %2214, i64 0, i64 0
  %.sub200.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %2213, i64 0, i64 0
  store i8* %malloccall47.i, i8** %.sub200.i, align 8, !noalias !133
  store i8 6, i8* %.sub203.i, align 1, !noalias !133
  %2217 = bitcast [8 x i64]* %2215 to <4 x i64>*
  store <4 x i64> <i64 1, i64 80, i64 14, i64 14>, <4 x i64>* %2217, align 8, !noalias !133
  %2218 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2213, i64 0, i64 1
  store i8* %malloccall22.i, i8** %2218, align 8, !noalias !133
  %2219 = getelementptr inbounds [3 x i8], [3 x i8]* %2216, i64 0, i64 1
  store i8 6, i8* %2219, align 1, !noalias !133
  %2220 = bitcast [3 x i64]* %2214 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %2220, align 16, !noalias !133
  %2221 = getelementptr inbounds [8 x i64], [8 x i64]* %2215, i64 0, i64 4
  %2222 = bitcast i64* %2221 to <4 x i64>*
  store <4 x i64> <i64 1, i64 184, i64 14, i64 14>, <4 x i64>* %2222, align 8, !noalias !133
  %2223 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2213, i64 0, i64 2
  %2224 = bitcast i8** %2223 to float**
  store float* %128, float** %2224, align 8, !noalias !133
  %2225 = getelementptr inbounds [3 x i8], [3 x i8]* %2216, i64 0, i64 2
  store i8 6, i8* %2225, align 1, !noalias !133
  %2226 = getelementptr inbounds [3 x i64], [3 x i64]* %2214, i64 0, i64 2
  store i64 0, i64* %2226, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub200.i, i64* nonnull %.sub201.i, i64* nonnull %.sub202.i, i8* nonnull %.sub203.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond363.preheader.i

cond351.preheader.i:                              ; preds = %cond351.preheader.i, %cond348.preheader.i
  %2227 = phi i64 [ 0, %cond348.preheader.i ], [ %2277, %cond351.preheader.i ]
  %2228 = mul nuw nsw i64 %2227, 14
  %2229 = add nuw nsw i64 %2228, %2210
  %2230 = getelementptr float, float* %284, i64 %2229
  %2231 = getelementptr float, float* %262, i64 %2229
  %2232 = bitcast float* %2230 to <8 x float>*
  %2233 = load <8 x float>, <8 x float>* %2232, align 4, !noalias !133
  %2234 = fadd <8 x float> %2233, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %2235 = fcmp olt <8 x float> %2234, zeroinitializer
  %2236 = select <8 x i1> %2235, <8 x float> zeroinitializer, <8 x float> %2234
  %2237 = fcmp ogt <8 x float> %2236, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2238 = select <8 x i1> %2237, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %2236
  %2239 = fmul <8 x float> %2233, %2238
  %2240 = fdiv <8 x float> %2239, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2241 = bitcast float* %2231 to <8 x float>*
  store <8 x float> %2240, <8 x float>* %2241, align 4, !noalias !133
  %2242 = add nuw nsw i64 %2211, %2228
  %2243 = getelementptr float, float* %284, i64 %2242
  %2244 = getelementptr float, float* %262, i64 %2242
  %2245 = bitcast float* %2243 to <4 x float>*
  %2246 = load <4 x float>, <4 x float>* %2245, align 4, !noalias !133
  %2247 = fadd <4 x float> %2246, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %2248 = fcmp olt <4 x float> %2247, zeroinitializer
  %2249 = select <4 x i1> %2248, <4 x float> zeroinitializer, <4 x float> %2247
  %2250 = fcmp ogt <4 x float> %2249, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2251 = select <4 x i1> %2250, <4 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <4 x float> %2249
  %2252 = fmul <4 x float> %2246, %2251
  %2253 = fdiv <4 x float> %2252, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2254 = bitcast float* %2244 to <4 x float>*
  store <4 x float> %2253, <4 x float>* %2254, align 4, !noalias !133
  %2255 = add nuw nsw i64 %2212, %2228
  %2256 = getelementptr float, float* %284, i64 %2255
  %2257 = load float, float* %2256, align 4, !noalias !133
  %2258 = fadd float %2257, 3.000000e+00
  %2259 = fcmp olt float %2258, 0.000000e+00
  %2260 = select i1 %2259, float 0.000000e+00, float %2258
  %2261 = fcmp ogt float %2260, 6.000000e+00
  %2262 = select i1 %2261, float 6.000000e+00, float %2260
  %2263 = fmul float %2257, %2262
  %2264 = fdiv float %2263, 6.000000e+00
  %2265 = getelementptr float, float* %262, i64 %2255
  store float %2264, float* %2265, align 4, !noalias !133
  %2266 = or i64 %2255, 1
  %2267 = getelementptr float, float* %284, i64 %2266
  %2268 = load float, float* %2267, align 4, !noalias !133
  %2269 = fadd float %2268, 3.000000e+00
  %2270 = fcmp olt float %2269, 0.000000e+00
  %2271 = select i1 %2270, float 0.000000e+00, float %2269
  %2272 = fcmp ogt float %2271, 6.000000e+00
  %2273 = select i1 %2272, float 6.000000e+00, float %2271
  %2274 = fmul float %2268, %2273
  %2275 = fdiv float %2274, 6.000000e+00
  %2276 = getelementptr float, float* %262, i64 %2266
  store float %2275, float* %2276, align 4, !noalias !133
  %2277 = add nuw nsw i64 %2227, 1
  %exitcond493.not.i = icmp eq i64 %2277, 14
  br i1 %exitcond493.not.i, label %exit350.i, label %cond351.preheader.i

exit350.i:                                        ; preds = %cond351.preheader.i
  %2278 = add nuw nsw i64 %2209, 1
  %exitcond494.not.i = icmp eq i64 %2278, 184
  br i1 %exitcond494.not.i, label %exit347.i, label %cond348.preheader.i

cond363.preheader.i:                              ; preds = %exit365.i, %exit347.i
  %2279 = phi i64 [ 0, %exit347.i ], [ %2370, %exit365.i ]
  %2280 = mul nuw nsw i64 %2279, 196
  %2281 = add nuw nsw i64 %2280, 8
  %2282 = add nuw nsw i64 %2280, 12
  br label %cond366.preheader.i

exit362.i:                                        ; preds = %exit365.i
  %2283 = alloca [3 x i8*], align 8
  %2284 = alloca [3 x i64], align 16
  %2285 = alloca [8 x i64], align 8
  %2286 = alloca [3 x i8], align 1
  %.sub208.i = getelementptr inbounds [3 x i8], [3 x i8]* %2286, i64 0, i64 0
  %.sub207.i = getelementptr inbounds [8 x i64], [8 x i64]* %2285, i64 0, i64 0
  %.sub206.i = getelementptr inbounds [3 x i64], [3 x i64]* %2284, i64 0, i64 0
  %.sub205.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %2283, i64 0, i64 0
  store i8* %malloccall18.i, i8** %.sub205.i, align 8, !noalias !133
  store i8 6, i8* %.sub208.i, align 1, !noalias !133
  %2287 = bitcast [8 x i64]* %2285 to <4 x i64>*
  store <4 x i64> <i64 1, i64 480, i64 14, i64 14>, <4 x i64>* %2287, align 8, !noalias !133
  %2288 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2283, i64 0, i64 1
  store i8* %malloccall19.i, i8** %2288, align 8, !noalias !133
  %2289 = getelementptr inbounds [3 x i8], [3 x i8]* %2286, i64 0, i64 1
  store i8 6, i8* %2289, align 1, !noalias !133
  %2290 = bitcast [3 x i64]* %2284 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %2290, align 16, !noalias !133
  %2291 = getelementptr inbounds [8 x i64], [8 x i64]* %2285, i64 0, i64 4
  %2292 = bitcast i64* %2291 to <4 x i64>*
  store <4 x i64> <i64 1, i64 80, i64 14, i64 14>, <4 x i64>* %2292, align 8, !noalias !133
  %2293 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2283, i64 0, i64 2
  %2294 = bitcast i8** %2293 to float**
  store float* %131, float** %2294, align 8, !noalias !133
  %2295 = getelementptr inbounds [3 x i8], [3 x i8]* %2286, i64 0, i64 2
  store i8 6, i8* %2295, align 1, !noalias !133
  %2296 = getelementptr inbounds [3 x i64], [3 x i64]* %2284, i64 0, i64 2
  store i64 0, i64* %2296, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub205.i, i64* nonnull %.sub206.i, i64* nonnull %.sub207.i, i8* nonnull %.sub208.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond378.preheader.i

cond366.preheader.i:                              ; preds = %cond366.preheader.i, %cond363.preheader.i
  %2297 = phi i64 [ 0, %cond363.preheader.i ], [ %2369, %cond366.preheader.i ]
  %2298 = mul nuw nsw i64 %2297, 14
  %2299 = add nuw nsw i64 %2298, %2280
  %2300 = getelementptr float, float* %283, i64 %2299
  %2301 = getelementptr float, float* %291, i64 %2299
  %2302 = getelementptr float, float* %260, i64 %2299
  %2303 = bitcast float* %2300 to <8 x float>*
  %2304 = load <8 x float>, <8 x float>* %2303, align 4, !noalias !133
  %2305 = bitcast float* %2301 to <8 x float>*
  %2306 = load <8 x float>, <8 x float>* %2305, align 4, !noalias !133
  %2307 = fadd <8 x float> %2304, %2306
  %2308 = bitcast float* %2302 to <8 x float>*
  store <8 x float> %2307, <8 x float>* %2308, align 4, !noalias !133
  %2309 = add nuw nsw i64 %2281, %2298
  %2310 = getelementptr float, float* %283, i64 %2309
  %2311 = getelementptr float, float* %291, i64 %2309
  %2312 = getelementptr float, float* %260, i64 %2309
  %2313 = bitcast float* %2310 to <4 x float>*
  %2314 = load <4 x float>, <4 x float>* %2313, align 4, !noalias !133
  %2315 = bitcast float* %2311 to <4 x float>*
  %2316 = load <4 x float>, <4 x float>* %2315, align 4, !noalias !133
  %2317 = fadd <4 x float> %2314, %2316
  %2318 = bitcast float* %2312 to <4 x float>*
  store <4 x float> %2317, <4 x float>* %2318, align 4, !noalias !133
  %2319 = add nuw nsw i64 %2282, %2298
  %2320 = getelementptr float, float* %283, i64 %2319
  %2321 = load float, float* %2320, align 4, !noalias !133
  %2322 = getelementptr float, float* %291, i64 %2319
  %2323 = load float, float* %2322, align 4, !noalias !133
  %2324 = fadd float %2321, %2323
  %2325 = getelementptr float, float* %260, i64 %2319
  store float %2324, float* %2325, align 4, !noalias !133
  %2326 = or i64 %2319, 1
  %2327 = getelementptr float, float* %283, i64 %2326
  %2328 = load float, float* %2327, align 4, !noalias !133
  %2329 = getelementptr float, float* %291, i64 %2326
  %2330 = load float, float* %2329, align 4, !noalias !133
  %2331 = fadd float %2328, %2330
  %2332 = getelementptr float, float* %260, i64 %2326
  store float %2331, float* %2332, align 4, !noalias !133
  %2333 = or i64 %2297, 1
  %2334 = mul nuw nsw i64 %2333, 14
  %2335 = add nuw nsw i64 %2334, %2280
  %2336 = getelementptr float, float* %283, i64 %2335
  %2337 = getelementptr float, float* %291, i64 %2335
  %2338 = getelementptr float, float* %260, i64 %2335
  %2339 = bitcast float* %2336 to <8 x float>*
  %2340 = load <8 x float>, <8 x float>* %2339, align 4, !noalias !133
  %2341 = bitcast float* %2337 to <8 x float>*
  %2342 = load <8 x float>, <8 x float>* %2341, align 4, !noalias !133
  %2343 = fadd <8 x float> %2340, %2342
  %2344 = bitcast float* %2338 to <8 x float>*
  store <8 x float> %2343, <8 x float>* %2344, align 4, !noalias !133
  %2345 = add nuw nsw i64 %2281, %2334
  %2346 = getelementptr float, float* %283, i64 %2345
  %2347 = getelementptr float, float* %291, i64 %2345
  %2348 = getelementptr float, float* %260, i64 %2345
  %2349 = bitcast float* %2346 to <4 x float>*
  %2350 = load <4 x float>, <4 x float>* %2349, align 4, !noalias !133
  %2351 = bitcast float* %2347 to <4 x float>*
  %2352 = load <4 x float>, <4 x float>* %2351, align 4, !noalias !133
  %2353 = fadd <4 x float> %2350, %2352
  %2354 = bitcast float* %2348 to <4 x float>*
  store <4 x float> %2353, <4 x float>* %2354, align 4, !noalias !133
  %2355 = add nuw nsw i64 %2282, %2334
  %2356 = getelementptr float, float* %283, i64 %2355
  %2357 = load float, float* %2356, align 4, !noalias !133
  %2358 = getelementptr float, float* %291, i64 %2355
  %2359 = load float, float* %2358, align 4, !noalias !133
  %2360 = fadd float %2357, %2359
  %2361 = getelementptr float, float* %260, i64 %2355
  store float %2360, float* %2361, align 4, !noalias !133
  %2362 = or i64 %2355, 1
  %2363 = getelementptr float, float* %283, i64 %2362
  %2364 = load float, float* %2363, align 4, !noalias !133
  %2365 = getelementptr float, float* %291, i64 %2362
  %2366 = load float, float* %2365, align 4, !noalias !133
  %2367 = fadd float %2364, %2366
  %2368 = getelementptr float, float* %260, i64 %2362
  store float %2367, float* %2368, align 4, !noalias !133
  %2369 = add nuw nsw i64 %2297, 2
  %exitcond489.not.1.i = icmp eq i64 %2369, 14
  br i1 %exitcond489.not.1.i, label %exit365.i, label %cond366.preheader.i

exit365.i:                                        ; preds = %cond366.preheader.i
  %2370 = add nuw nsw i64 %2279, 1
  %exitcond490.not.i = icmp eq i64 %2370, 80
  br i1 %exitcond490.not.i, label %exit362.i, label %cond363.preheader.i

cond378.preheader.i:                              ; preds = %exit380.i, %exit362.i
  %2371 = phi i64 [ 0, %exit362.i ], [ %2440, %exit380.i ]
  %2372 = mul nuw nsw i64 %2371, 196
  %2373 = add nuw nsw i64 %2372, 8
  %2374 = add nuw nsw i64 %2372, 12
  br label %cond381.preheader.i

exit377.i:                                        ; preds = %exit380.i
  %2375 = alloca [3 x i8*], align 8
  %2376 = alloca [3 x i64], align 16
  %2377 = alloca [8 x i64], align 8
  %2378 = alloca [3 x i8], align 1
  %.sub213.i = getelementptr inbounds [3 x i8], [3 x i8]* %2378, i64 0, i64 0
  %.sub212.i = getelementptr inbounds [8 x i64], [8 x i64]* %2377, i64 0, i64 0
  %.sub211.i = getelementptr inbounds [3 x i64], [3 x i64]* %2376, i64 0, i64 0
  %.sub210.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %2375, i64 0, i64 0
  store i8* %malloccall17.i, i8** %.sub210.i, align 8, !noalias !133
  store i8 6, i8* %.sub213.i, align 1, !noalias !133
  %2379 = bitcast [8 x i64]* %2377 to <4 x i64>*
  store <4 x i64> <i64 1, i64 480, i64 14, i64 14>, <4 x i64>* %2379, align 8, !noalias !133
  %2380 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2375, i64 0, i64 1
  store i8* %malloccall62.i, i8** %2380, align 8, !noalias !133
  %2381 = getelementptr inbounds [3 x i8], [3 x i8]* %2378, i64 0, i64 1
  store i8 6, i8* %2381, align 1, !noalias !133
  %2382 = bitcast [3 x i64]* %2376 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %2382, align 16, !noalias !133
  %2383 = getelementptr inbounds [8 x i64], [8 x i64]* %2377, i64 0, i64 4
  %2384 = bitcast i64* %2383 to <4 x i64>*
  store <4 x i64> <i64 1, i64 480, i64 14, i64 14>, <4 x i64>* %2384, align 8, !noalias !133
  %2385 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2375, i64 0, i64 2
  %2386 = bitcast i8** %2385 to float**
  store float* %134, float** %2386, align 8, !noalias !133
  %2387 = getelementptr inbounds [3 x i8], [3 x i8]* %2378, i64 0, i64 2
  store i8 6, i8* %2387, align 1, !noalias !133
  %2388 = getelementptr inbounds [3 x i64], [3 x i64]* %2376, i64 0, i64 2
  store i64 0, i64* %2388, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub210.i, i64* nonnull %.sub211.i, i64* nonnull %.sub212.i, i8* nonnull %.sub213.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond393.preheader.i

cond381.preheader.i:                              ; preds = %cond381.preheader.i, %cond378.preheader.i
  %2389 = phi i64 [ 0, %cond378.preheader.i ], [ %2439, %cond381.preheader.i ]
  %2390 = mul nuw nsw i64 %2389, 14
  %2391 = add nuw nsw i64 %2390, %2372
  %2392 = getelementptr float, float* %259, i64 %2391
  %2393 = getelementptr float, float* %294, i64 %2391
  %2394 = bitcast float* %2392 to <8 x float>*
  %2395 = load <8 x float>, <8 x float>* %2394, align 4, !noalias !133
  %2396 = fadd <8 x float> %2395, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %2397 = fcmp olt <8 x float> %2396, zeroinitializer
  %2398 = select <8 x i1> %2397, <8 x float> zeroinitializer, <8 x float> %2396
  %2399 = fcmp ogt <8 x float> %2398, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2400 = select <8 x i1> %2399, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %2398
  %2401 = fmul <8 x float> %2395, %2400
  %2402 = fdiv <8 x float> %2401, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2403 = bitcast float* %2393 to <8 x float>*
  store <8 x float> %2402, <8 x float>* %2403, align 4, !noalias !133
  %2404 = add nuw nsw i64 %2373, %2390
  %2405 = getelementptr float, float* %259, i64 %2404
  %2406 = getelementptr float, float* %294, i64 %2404
  %2407 = bitcast float* %2405 to <4 x float>*
  %2408 = load <4 x float>, <4 x float>* %2407, align 4, !noalias !133
  %2409 = fadd <4 x float> %2408, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %2410 = fcmp olt <4 x float> %2409, zeroinitializer
  %2411 = select <4 x i1> %2410, <4 x float> zeroinitializer, <4 x float> %2409
  %2412 = fcmp ogt <4 x float> %2411, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2413 = select <4 x i1> %2412, <4 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <4 x float> %2411
  %2414 = fmul <4 x float> %2408, %2413
  %2415 = fdiv <4 x float> %2414, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2416 = bitcast float* %2406 to <4 x float>*
  store <4 x float> %2415, <4 x float>* %2416, align 4, !noalias !133
  %2417 = add nuw nsw i64 %2374, %2390
  %2418 = getelementptr float, float* %259, i64 %2417
  %2419 = load float, float* %2418, align 4, !noalias !133
  %2420 = fadd float %2419, 3.000000e+00
  %2421 = fcmp olt float %2420, 0.000000e+00
  %2422 = select i1 %2421, float 0.000000e+00, float %2420
  %2423 = fcmp ogt float %2422, 6.000000e+00
  %2424 = select i1 %2423, float 6.000000e+00, float %2422
  %2425 = fmul float %2419, %2424
  %2426 = fdiv float %2425, 6.000000e+00
  %2427 = getelementptr float, float* %294, i64 %2417
  store float %2426, float* %2427, align 4, !noalias !133
  %2428 = or i64 %2417, 1
  %2429 = getelementptr float, float* %259, i64 %2428
  %2430 = load float, float* %2429, align 4, !noalias !133
  %2431 = fadd float %2430, 3.000000e+00
  %2432 = fcmp olt float %2431, 0.000000e+00
  %2433 = select i1 %2432, float 0.000000e+00, float %2431
  %2434 = fcmp ogt float %2433, 6.000000e+00
  %2435 = select i1 %2434, float 6.000000e+00, float %2433
  %2436 = fmul float %2430, %2435
  %2437 = fdiv float %2436, 6.000000e+00
  %2438 = getelementptr float, float* %294, i64 %2428
  store float %2437, float* %2438, align 4, !noalias !133
  %2439 = add nuw nsw i64 %2389, 1
  %exitcond485.not.i = icmp eq i64 %2439, 14
  br i1 %exitcond485.not.i, label %exit380.i, label %cond381.preheader.i

exit380.i:                                        ; preds = %cond381.preheader.i
  %2440 = add nuw nsw i64 %2371, 1
  %exitcond486.not.i = icmp eq i64 %2440, 480
  br i1 %exitcond486.not.i, label %exit377.i, label %cond378.preheader.i

cond393.preheader.i:                              ; preds = %exit395.i, %exit377.i
  %2441 = phi i64 [ 0, %exit377.i ], [ %2536, %exit395.i ]
  %2442 = mul nuw nsw i64 %2441, 196
  %2443 = add nuw nsw i64 %2442, 8
  %2444 = add nuw nsw i64 %2442, 12
  br label %cond396.preheader.i

exit392.i:                                        ; preds = %exit395.i
  %2445 = alloca [2 x i8*], align 8
  %2446 = alloca <2 x i64>, align 16
  %2447 = alloca [8 x i64], align 8
  %2448 = alloca [2 x i8], align 1
  %2449 = alloca <2 x i64>, align 16
  %.sub219.i = getelementptr inbounds <2 x i64>, <2 x i64>* %2449, i64 0, i64 0
  %.sub218.i = getelementptr inbounds [2 x i8], [2 x i8]* %2448, i64 0, i64 0
  %.sub217.i = getelementptr inbounds [8 x i64], [8 x i64]* %2447, i64 0, i64 0
  %.sub216.i = getelementptr inbounds <2 x i64>, <2 x i64>* %2446, i64 0, i64 0
  %.sub215.i = getelementptr inbounds [2 x i8*], [2 x i8*]* %2445, i64 0, i64 0
  store i8* %malloccall21.i, i8** %.sub215.i, align 8, !noalias !133
  store i8 6, i8* %.sub218.i, align 1, !noalias !133
  %2450 = bitcast [8 x i64]* %2447 to <4 x i64>*
  store <4 x i64> <i64 1, i64 480, i64 1, i64 1>, <4 x i64>* %2450, align 8, !noalias !133
  %2451 = getelementptr inbounds [2 x i8*], [2 x i8*]* %2445, i64 0, i64 1
  store i8* %malloccall79.i, i8** %2451, align 8, !noalias !133
  %2452 = getelementptr inbounds [2 x i8], [2 x i8]* %2448, i64 0, i64 1
  store i8 6, i8* %2452, align 1, !noalias !133
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %2446, align 16, !noalias !133
  %2453 = getelementptr inbounds [8 x i64], [8 x i64]* %2447, i64 0, i64 4
  %2454 = bitcast i64* %2453 to <4 x i64>*
  store <4 x i64> <i64 1, i64 480, i64 14, i64 14>, <4 x i64>* %2454, align 8, !noalias !133
  store <2 x i64> <i64 1, i64 1>, <2 x i64>* %2449, align 16, !noalias !133
  call void @nnc_aten_adaptive_avg_pool2d(i64 2, i8** nonnull %.sub215.i, i64* nonnull %.sub216.i, i64* nonnull %.sub217.i, i8* nonnull %.sub218.i, i64 2, i64* nonnull %.sub219.i) #0, !noalias !135
  %2455 = alloca [3 x i8*], align 8
  %2456 = alloca [3 x i64], align 16
  %2457 = alloca [8 x i64], align 8
  %2458 = alloca [3 x i8], align 1
  %.sub223.i = getelementptr inbounds [3 x i8], [3 x i8]* %2458, i64 0, i64 0
  %.sub222.i = getelementptr inbounds [8 x i64], [8 x i64]* %2457, i64 0, i64 0
  %.sub221.i = getelementptr inbounds [3 x i64], [3 x i64]* %2456, i64 0, i64 0
  %.sub220.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %2455, i64 0, i64 0
  %2459 = bitcast [3 x i8*]* %2455 to [480 x float]**
  store [480 x float]* %2, [480 x float]** %2459, align 8, !noalias !133
  store i8 6, i8* %.sub223.i, align 1, !noalias !133
  %2460 = bitcast [8 x i64]* %2457 to <4 x i64>*
  store <4 x i64> <i64 1, i64 120, i64 1, i64 1>, <4 x i64>* %2460, align 8, !noalias !133
  %2461 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2455, i64 0, i64 1
  store i8* %malloccall21.i, i8** %2461, align 8, !noalias !133
  %2462 = getelementptr inbounds [3 x i8], [3 x i8]* %2458, i64 0, i64 1
  store i8 6, i8* %2462, align 1, !noalias !133
  %2463 = bitcast [3 x i64]* %2456 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %2463, align 16, !noalias !133
  %2464 = getelementptr inbounds [8 x i64], [8 x i64]* %2457, i64 0, i64 4
  %2465 = bitcast i64* %2464 to <4 x i64>*
  store <4 x i64> <i64 1, i64 480, i64 1, i64 1>, <4 x i64>* %2465, align 8, !noalias !133
  %2466 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2455, i64 0, i64 2
  %2467 = bitcast i8** %2466 to float**
  store float* %137, float** %2467, align 8, !noalias !133
  %2468 = getelementptr inbounds [3 x i8], [3 x i8]* %2458, i64 0, i64 2
  store i8 6, i8* %2468, align 1, !noalias !133
  %2469 = getelementptr inbounds [3 x i64], [3 x i64]* %2456, i64 0, i64 2
  store i64 0, i64* %2469, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub220.i, i64* nonnull %.sub221.i, i64* nonnull %.sub222.i, i8* nonnull %.sub223.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %2470 = alloca [3 x i8*], align 8
  %2471 = alloca [3 x i64], align 16
  %2472 = alloca [8 x i64], align 8
  %2473 = alloca [3 x i8], align 1
  %.sub228.i = getelementptr inbounds [3 x i8], [3 x i8]* %2473, i64 0, i64 0
  %.sub227.i = getelementptr inbounds [8 x i64], [8 x i64]* %2472, i64 0, i64 0
  %.sub226.i = getelementptr inbounds [3 x i64], [3 x i64]* %2471, i64 0, i64 0
  %.sub225.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %2470, i64 0, i64 0
  store i8* %malloccall15.i, i8** %.sub225.i, align 8, !noalias !133
  store i8 6, i8* %.sub228.i, align 1, !noalias !133
  %2474 = bitcast [8 x i64]* %2472 to <4 x i64>*
  store <4 x i64> <i64 1, i64 480, i64 1, i64 1>, <4 x i64>* %2474, align 8, !noalias !133
  %2475 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2470, i64 0, i64 1
  %2476 = bitcast i8** %2475 to [480 x float]**
  store [480 x float]* %2, [480 x float]** %2476, align 8, !noalias !133
  %2477 = getelementptr inbounds [3 x i8], [3 x i8]* %2473, i64 0, i64 1
  store i8 6, i8* %2477, align 1, !noalias !133
  %2478 = bitcast [3 x i64]* %2471 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %2478, align 16, !noalias !133
  %2479 = getelementptr inbounds [8 x i64], [8 x i64]* %2472, i64 0, i64 4
  %2480 = bitcast i64* %2479 to <4 x i64>*
  store <4 x i64> <i64 1, i64 120, i64 1, i64 1>, <4 x i64>* %2480, align 8, !noalias !133
  %2481 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2470, i64 0, i64 2
  %2482 = bitcast i8** %2481 to float**
  store float* %140, float** %2482, align 8, !noalias !133
  %2483 = getelementptr inbounds [3 x i8], [3 x i8]* %2473, i64 0, i64 2
  store i8 6, i8* %2483, align 1, !noalias !133
  %2484 = getelementptr inbounds [3 x i64], [3 x i64]* %2471, i64 0, i64 2
  store i64 0, i64* %2484, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub225.i, i64* nonnull %.sub226.i, i64* nonnull %.sub227.i, i8* nonnull %.sub228.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond408.preheader.i

cond396.preheader.i:                              ; preds = %cond396.preheader.i, %cond393.preheader.i
  %2485 = phi i64 [ 0, %cond393.preheader.i ], [ %2535, %cond396.preheader.i ]
  %2486 = mul nuw nsw i64 %2485, 14
  %2487 = add nuw nsw i64 %2486, %2442
  %2488 = getelementptr float, float* %258, i64 %2487
  %2489 = getelementptr float, float* %305, i64 %2487
  %2490 = bitcast float* %2488 to <8 x float>*
  %2491 = load <8 x float>, <8 x float>* %2490, align 4, !noalias !133
  %2492 = fadd <8 x float> %2491, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %2493 = fcmp olt <8 x float> %2492, zeroinitializer
  %2494 = select <8 x i1> %2493, <8 x float> zeroinitializer, <8 x float> %2492
  %2495 = fcmp ogt <8 x float> %2494, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2496 = select <8 x i1> %2495, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %2494
  %2497 = fmul <8 x float> %2491, %2496
  %2498 = fdiv <8 x float> %2497, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2499 = bitcast float* %2489 to <8 x float>*
  store <8 x float> %2498, <8 x float>* %2499, align 4, !noalias !133
  %2500 = add nuw nsw i64 %2443, %2486
  %2501 = getelementptr float, float* %258, i64 %2500
  %2502 = getelementptr float, float* %305, i64 %2500
  %2503 = bitcast float* %2501 to <4 x float>*
  %2504 = load <4 x float>, <4 x float>* %2503, align 4, !noalias !133
  %2505 = fadd <4 x float> %2504, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %2506 = fcmp olt <4 x float> %2505, zeroinitializer
  %2507 = select <4 x i1> %2506, <4 x float> zeroinitializer, <4 x float> %2505
  %2508 = fcmp ogt <4 x float> %2507, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2509 = select <4 x i1> %2508, <4 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <4 x float> %2507
  %2510 = fmul <4 x float> %2504, %2509
  %2511 = fdiv <4 x float> %2510, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2512 = bitcast float* %2502 to <4 x float>*
  store <4 x float> %2511, <4 x float>* %2512, align 4, !noalias !133
  %2513 = add nuw nsw i64 %2444, %2486
  %2514 = getelementptr float, float* %258, i64 %2513
  %2515 = load float, float* %2514, align 4, !noalias !133
  %2516 = fadd float %2515, 3.000000e+00
  %2517 = fcmp olt float %2516, 0.000000e+00
  %2518 = select i1 %2517, float 0.000000e+00, float %2516
  %2519 = fcmp ogt float %2518, 6.000000e+00
  %2520 = select i1 %2519, float 6.000000e+00, float %2518
  %2521 = fmul float %2515, %2520
  %2522 = fdiv float %2521, 6.000000e+00
  %2523 = getelementptr float, float* %305, i64 %2513
  store float %2522, float* %2523, align 4, !noalias !133
  %2524 = or i64 %2513, 1
  %2525 = getelementptr float, float* %258, i64 %2524
  %2526 = load float, float* %2525, align 4, !noalias !133
  %2527 = fadd float %2526, 3.000000e+00
  %2528 = fcmp olt float %2527, 0.000000e+00
  %2529 = select i1 %2528, float 0.000000e+00, float %2527
  %2530 = fcmp ogt float %2529, 6.000000e+00
  %2531 = select i1 %2530, float 6.000000e+00, float %2529
  %2532 = fmul float %2526, %2531
  %2533 = fdiv float %2532, 6.000000e+00
  %2534 = getelementptr float, float* %305, i64 %2524
  store float %2533, float* %2534, align 4, !noalias !133
  %2535 = add nuw nsw i64 %2485, 1
  %exitcond481.not.i = icmp eq i64 %2535, 14
  br i1 %exitcond481.not.i, label %exit395.i, label %cond396.preheader.i

exit395.i:                                        ; preds = %cond396.preheader.i
  %2536 = add nuw nsw i64 %2441, 1
  %exitcond482.not.i = icmp eq i64 %2536, 480
  br i1 %exitcond482.not.i, label %exit392.i, label %cond393.preheader.i

cond408.preheader.i:                              ; preds = %cond408.preheader.i, %exit392.i
  %2537 = phi i64 [ 0, %exit392.i ], [ %2885, %cond408.preheader.i ]
  %2538 = mul nuw nsw i64 %2537, 196
  %2539 = getelementptr float, float* %256, i64 %2537
  %2540 = load float, float* %2539, align 4, !noalias !133
  %2541 = fadd float %2540, 3.000000e+00
  %2542 = fcmp olt float %2541, 0.000000e+00
  %2543 = select i1 %2542, float 0.000000e+00, float %2541
  %2544 = fcmp ogt float %2543, 6.000000e+00
  %.op396.i = fdiv float %2541, 6.000000e+00
  %.op395.i = select i1 %2542, float 0.000000e+00, float %.op396.i
  %2545 = select i1 %2544, float 1.000000e+00, float %.op395.i
  %2546 = add nuw nsw i64 %2538, 8
  %2547 = add nuw nsw i64 %2538, 12
  %2548 = insertelement <8 x float> poison, float %2545, i32 0
  %2549 = shufflevector <8 x float> %2548, <8 x float> undef, <8 x i32> zeroinitializer
  %2550 = insertelement <4 x float> poison, float %2545, i32 0
  %2551 = shufflevector <4 x float> %2550, <4 x float> undef, <4 x i32> zeroinitializer
  %2552 = getelementptr float, float* %305, i64 %2538
  %2553 = getelementptr float, float* %255, i64 %2538
  %2554 = bitcast float* %2552 to <8 x float>*
  %2555 = load <8 x float>, <8 x float>* %2554, align 4, !noalias !133
  %2556 = fmul <8 x float> %2555, %2549
  %2557 = bitcast float* %2553 to <8 x float>*
  store <8 x float> %2556, <8 x float>* %2557, align 4, !noalias !133
  %2558 = getelementptr float, float* %305, i64 %2546
  %2559 = getelementptr float, float* %255, i64 %2546
  %2560 = bitcast float* %2558 to <4 x float>*
  %2561 = load <4 x float>, <4 x float>* %2560, align 4, !noalias !133
  %2562 = fmul <4 x float> %2561, %2551
  %2563 = bitcast float* %2559 to <4 x float>*
  store <4 x float> %2562, <4 x float>* %2563, align 4, !noalias !133
  %2564 = getelementptr float, float* %305, i64 %2547
  %2565 = load float, float* %2564, align 4, !noalias !133
  %2566 = fmul float %2565, %2545
  %2567 = getelementptr float, float* %255, i64 %2547
  store float %2566, float* %2567, align 4, !noalias !133
  %2568 = or i64 %2547, 1
  %2569 = getelementptr float, float* %305, i64 %2568
  %2570 = load float, float* %2569, align 4, !noalias !133
  %2571 = fmul float %2570, %2545
  %2572 = getelementptr float, float* %255, i64 %2568
  store float %2571, float* %2572, align 4, !noalias !133
  %2573 = add nuw nsw i64 %2538, 14
  %2574 = getelementptr float, float* %305, i64 %2573
  %2575 = getelementptr float, float* %255, i64 %2573
  %2576 = bitcast float* %2574 to <8 x float>*
  %2577 = load <8 x float>, <8 x float>* %2576, align 4, !noalias !133
  %2578 = fmul <8 x float> %2577, %2549
  %2579 = bitcast float* %2575 to <8 x float>*
  store <8 x float> %2578, <8 x float>* %2579, align 4, !noalias !133
  %2580 = add nuw nsw i64 %2538, 22
  %2581 = getelementptr float, float* %305, i64 %2580
  %2582 = getelementptr float, float* %255, i64 %2580
  %2583 = bitcast float* %2581 to <4 x float>*
  %2584 = load <4 x float>, <4 x float>* %2583, align 4, !noalias !133
  %2585 = fmul <4 x float> %2584, %2551
  %2586 = bitcast float* %2582 to <4 x float>*
  store <4 x float> %2585, <4 x float>* %2586, align 4, !noalias !133
  %2587 = add nuw nsw i64 %2538, 26
  %2588 = getelementptr float, float* %305, i64 %2587
  %2589 = load float, float* %2588, align 4, !noalias !133
  %2590 = fmul float %2589, %2545
  %2591 = getelementptr float, float* %255, i64 %2587
  store float %2590, float* %2591, align 4, !noalias !133
  %2592 = or i64 %2587, 1
  %2593 = getelementptr float, float* %305, i64 %2592
  %2594 = load float, float* %2593, align 4, !noalias !133
  %2595 = fmul float %2594, %2545
  %2596 = getelementptr float, float* %255, i64 %2592
  store float %2595, float* %2596, align 4, !noalias !133
  %2597 = add nuw nsw i64 %2538, 28
  %2598 = getelementptr float, float* %305, i64 %2597
  %2599 = getelementptr float, float* %255, i64 %2597
  %2600 = bitcast float* %2598 to <8 x float>*
  %2601 = load <8 x float>, <8 x float>* %2600, align 4, !noalias !133
  %2602 = fmul <8 x float> %2601, %2549
  %2603 = bitcast float* %2599 to <8 x float>*
  store <8 x float> %2602, <8 x float>* %2603, align 4, !noalias !133
  %2604 = add nuw nsw i64 %2538, 36
  %2605 = getelementptr float, float* %305, i64 %2604
  %2606 = getelementptr float, float* %255, i64 %2604
  %2607 = bitcast float* %2605 to <4 x float>*
  %2608 = load <4 x float>, <4 x float>* %2607, align 4, !noalias !133
  %2609 = fmul <4 x float> %2608, %2551
  %2610 = bitcast float* %2606 to <4 x float>*
  store <4 x float> %2609, <4 x float>* %2610, align 4, !noalias !133
  %2611 = add nuw nsw i64 %2538, 40
  %2612 = getelementptr float, float* %305, i64 %2611
  %2613 = load float, float* %2612, align 4, !noalias !133
  %2614 = fmul float %2613, %2545
  %2615 = getelementptr float, float* %255, i64 %2611
  store float %2614, float* %2615, align 4, !noalias !133
  %2616 = or i64 %2611, 1
  %2617 = getelementptr float, float* %305, i64 %2616
  %2618 = load float, float* %2617, align 4, !noalias !133
  %2619 = fmul float %2618, %2545
  %2620 = getelementptr float, float* %255, i64 %2616
  store float %2619, float* %2620, align 4, !noalias !133
  %2621 = add nuw nsw i64 %2538, 42
  %2622 = getelementptr float, float* %305, i64 %2621
  %2623 = getelementptr float, float* %255, i64 %2621
  %2624 = bitcast float* %2622 to <8 x float>*
  %2625 = load <8 x float>, <8 x float>* %2624, align 4, !noalias !133
  %2626 = fmul <8 x float> %2625, %2549
  %2627 = bitcast float* %2623 to <8 x float>*
  store <8 x float> %2626, <8 x float>* %2627, align 4, !noalias !133
  %2628 = add nuw nsw i64 %2538, 50
  %2629 = getelementptr float, float* %305, i64 %2628
  %2630 = getelementptr float, float* %255, i64 %2628
  %2631 = bitcast float* %2629 to <4 x float>*
  %2632 = load <4 x float>, <4 x float>* %2631, align 4, !noalias !133
  %2633 = fmul <4 x float> %2632, %2551
  %2634 = bitcast float* %2630 to <4 x float>*
  store <4 x float> %2633, <4 x float>* %2634, align 4, !noalias !133
  %2635 = add nuw nsw i64 %2538, 54
  %2636 = getelementptr float, float* %305, i64 %2635
  %2637 = load float, float* %2636, align 4, !noalias !133
  %2638 = fmul float %2637, %2545
  %2639 = getelementptr float, float* %255, i64 %2635
  store float %2638, float* %2639, align 4, !noalias !133
  %2640 = or i64 %2635, 1
  %2641 = getelementptr float, float* %305, i64 %2640
  %2642 = load float, float* %2641, align 4, !noalias !133
  %2643 = fmul float %2642, %2545
  %2644 = getelementptr float, float* %255, i64 %2640
  store float %2643, float* %2644, align 4, !noalias !133
  %2645 = add nuw nsw i64 %2538, 56
  %2646 = getelementptr float, float* %305, i64 %2645
  %2647 = getelementptr float, float* %255, i64 %2645
  %2648 = bitcast float* %2646 to <8 x float>*
  %2649 = load <8 x float>, <8 x float>* %2648, align 4, !noalias !133
  %2650 = fmul <8 x float> %2649, %2549
  %2651 = bitcast float* %2647 to <8 x float>*
  store <8 x float> %2650, <8 x float>* %2651, align 4, !noalias !133
  %2652 = add nuw nsw i64 %2538, 64
  %2653 = getelementptr float, float* %305, i64 %2652
  %2654 = getelementptr float, float* %255, i64 %2652
  %2655 = bitcast float* %2653 to <4 x float>*
  %2656 = load <4 x float>, <4 x float>* %2655, align 4, !noalias !133
  %2657 = fmul <4 x float> %2656, %2551
  %2658 = bitcast float* %2654 to <4 x float>*
  store <4 x float> %2657, <4 x float>* %2658, align 4, !noalias !133
  %2659 = add nuw nsw i64 %2538, 68
  %2660 = getelementptr float, float* %305, i64 %2659
  %2661 = load float, float* %2660, align 4, !noalias !133
  %2662 = fmul float %2661, %2545
  %2663 = getelementptr float, float* %255, i64 %2659
  store float %2662, float* %2663, align 4, !noalias !133
  %2664 = or i64 %2659, 1
  %2665 = getelementptr float, float* %305, i64 %2664
  %2666 = load float, float* %2665, align 4, !noalias !133
  %2667 = fmul float %2666, %2545
  %2668 = getelementptr float, float* %255, i64 %2664
  store float %2667, float* %2668, align 4, !noalias !133
  %2669 = add nuw nsw i64 %2538, 70
  %2670 = getelementptr float, float* %305, i64 %2669
  %2671 = getelementptr float, float* %255, i64 %2669
  %2672 = bitcast float* %2670 to <8 x float>*
  %2673 = load <8 x float>, <8 x float>* %2672, align 4, !noalias !133
  %2674 = fmul <8 x float> %2673, %2549
  %2675 = bitcast float* %2671 to <8 x float>*
  store <8 x float> %2674, <8 x float>* %2675, align 4, !noalias !133
  %2676 = add nuw nsw i64 %2538, 78
  %2677 = getelementptr float, float* %305, i64 %2676
  %2678 = getelementptr float, float* %255, i64 %2676
  %2679 = bitcast float* %2677 to <4 x float>*
  %2680 = load <4 x float>, <4 x float>* %2679, align 4, !noalias !133
  %2681 = fmul <4 x float> %2680, %2551
  %2682 = bitcast float* %2678 to <4 x float>*
  store <4 x float> %2681, <4 x float>* %2682, align 4, !noalias !133
  %2683 = add nuw nsw i64 %2538, 82
  %2684 = getelementptr float, float* %305, i64 %2683
  %2685 = load float, float* %2684, align 4, !noalias !133
  %2686 = fmul float %2685, %2545
  %2687 = getelementptr float, float* %255, i64 %2683
  store float %2686, float* %2687, align 4, !noalias !133
  %2688 = or i64 %2683, 1
  %2689 = getelementptr float, float* %305, i64 %2688
  %2690 = load float, float* %2689, align 4, !noalias !133
  %2691 = fmul float %2690, %2545
  %2692 = getelementptr float, float* %255, i64 %2688
  store float %2691, float* %2692, align 4, !noalias !133
  %2693 = add nuw nsw i64 %2538, 84
  %2694 = getelementptr float, float* %305, i64 %2693
  %2695 = getelementptr float, float* %255, i64 %2693
  %2696 = bitcast float* %2694 to <8 x float>*
  %2697 = load <8 x float>, <8 x float>* %2696, align 4, !noalias !133
  %2698 = fmul <8 x float> %2697, %2549
  %2699 = bitcast float* %2695 to <8 x float>*
  store <8 x float> %2698, <8 x float>* %2699, align 4, !noalias !133
  %2700 = add nuw nsw i64 %2538, 92
  %2701 = getelementptr float, float* %305, i64 %2700
  %2702 = getelementptr float, float* %255, i64 %2700
  %2703 = bitcast float* %2701 to <4 x float>*
  %2704 = load <4 x float>, <4 x float>* %2703, align 4, !noalias !133
  %2705 = fmul <4 x float> %2704, %2551
  %2706 = bitcast float* %2702 to <4 x float>*
  store <4 x float> %2705, <4 x float>* %2706, align 4, !noalias !133
  %2707 = add nuw nsw i64 %2538, 96
  %2708 = getelementptr float, float* %305, i64 %2707
  %2709 = load float, float* %2708, align 4, !noalias !133
  %2710 = fmul float %2709, %2545
  %2711 = getelementptr float, float* %255, i64 %2707
  store float %2710, float* %2711, align 4, !noalias !133
  %2712 = or i64 %2707, 1
  %2713 = getelementptr float, float* %305, i64 %2712
  %2714 = load float, float* %2713, align 4, !noalias !133
  %2715 = fmul float %2714, %2545
  %2716 = getelementptr float, float* %255, i64 %2712
  store float %2715, float* %2716, align 4, !noalias !133
  %2717 = add nuw nsw i64 %2538, 98
  %2718 = getelementptr float, float* %305, i64 %2717
  %2719 = getelementptr float, float* %255, i64 %2717
  %2720 = bitcast float* %2718 to <8 x float>*
  %2721 = load <8 x float>, <8 x float>* %2720, align 4, !noalias !133
  %2722 = fmul <8 x float> %2721, %2549
  %2723 = bitcast float* %2719 to <8 x float>*
  store <8 x float> %2722, <8 x float>* %2723, align 4, !noalias !133
  %2724 = add nuw nsw i64 %2538, 106
  %2725 = getelementptr float, float* %305, i64 %2724
  %2726 = getelementptr float, float* %255, i64 %2724
  %2727 = bitcast float* %2725 to <4 x float>*
  %2728 = load <4 x float>, <4 x float>* %2727, align 4, !noalias !133
  %2729 = fmul <4 x float> %2728, %2551
  %2730 = bitcast float* %2726 to <4 x float>*
  store <4 x float> %2729, <4 x float>* %2730, align 4, !noalias !133
  %2731 = add nuw nsw i64 %2538, 110
  %2732 = getelementptr float, float* %305, i64 %2731
  %2733 = load float, float* %2732, align 4, !noalias !133
  %2734 = fmul float %2733, %2545
  %2735 = getelementptr float, float* %255, i64 %2731
  store float %2734, float* %2735, align 4, !noalias !133
  %2736 = or i64 %2731, 1
  %2737 = getelementptr float, float* %305, i64 %2736
  %2738 = load float, float* %2737, align 4, !noalias !133
  %2739 = fmul float %2738, %2545
  %2740 = getelementptr float, float* %255, i64 %2736
  store float %2739, float* %2740, align 4, !noalias !133
  %2741 = add nuw nsw i64 %2538, 112
  %2742 = getelementptr float, float* %305, i64 %2741
  %2743 = getelementptr float, float* %255, i64 %2741
  %2744 = bitcast float* %2742 to <8 x float>*
  %2745 = load <8 x float>, <8 x float>* %2744, align 4, !noalias !133
  %2746 = fmul <8 x float> %2745, %2549
  %2747 = bitcast float* %2743 to <8 x float>*
  store <8 x float> %2746, <8 x float>* %2747, align 4, !noalias !133
  %2748 = add nuw nsw i64 %2538, 120
  %2749 = getelementptr float, float* %305, i64 %2748
  %2750 = getelementptr float, float* %255, i64 %2748
  %2751 = bitcast float* %2749 to <4 x float>*
  %2752 = load <4 x float>, <4 x float>* %2751, align 4, !noalias !133
  %2753 = fmul <4 x float> %2752, %2551
  %2754 = bitcast float* %2750 to <4 x float>*
  store <4 x float> %2753, <4 x float>* %2754, align 4, !noalias !133
  %2755 = add nuw nsw i64 %2538, 124
  %2756 = getelementptr float, float* %305, i64 %2755
  %2757 = load float, float* %2756, align 4, !noalias !133
  %2758 = fmul float %2757, %2545
  %2759 = getelementptr float, float* %255, i64 %2755
  store float %2758, float* %2759, align 4, !noalias !133
  %2760 = or i64 %2755, 1
  %2761 = getelementptr float, float* %305, i64 %2760
  %2762 = load float, float* %2761, align 4, !noalias !133
  %2763 = fmul float %2762, %2545
  %2764 = getelementptr float, float* %255, i64 %2760
  store float %2763, float* %2764, align 4, !noalias !133
  %2765 = add nuw nsw i64 %2538, 126
  %2766 = getelementptr float, float* %305, i64 %2765
  %2767 = getelementptr float, float* %255, i64 %2765
  %2768 = bitcast float* %2766 to <8 x float>*
  %2769 = load <8 x float>, <8 x float>* %2768, align 4, !noalias !133
  %2770 = fmul <8 x float> %2769, %2549
  %2771 = bitcast float* %2767 to <8 x float>*
  store <8 x float> %2770, <8 x float>* %2771, align 4, !noalias !133
  %2772 = add nuw nsw i64 %2538, 134
  %2773 = getelementptr float, float* %305, i64 %2772
  %2774 = getelementptr float, float* %255, i64 %2772
  %2775 = bitcast float* %2773 to <4 x float>*
  %2776 = load <4 x float>, <4 x float>* %2775, align 4, !noalias !133
  %2777 = fmul <4 x float> %2776, %2551
  %2778 = bitcast float* %2774 to <4 x float>*
  store <4 x float> %2777, <4 x float>* %2778, align 4, !noalias !133
  %2779 = add nuw nsw i64 %2538, 138
  %2780 = getelementptr float, float* %305, i64 %2779
  %2781 = load float, float* %2780, align 4, !noalias !133
  %2782 = fmul float %2781, %2545
  %2783 = getelementptr float, float* %255, i64 %2779
  store float %2782, float* %2783, align 4, !noalias !133
  %2784 = or i64 %2779, 1
  %2785 = getelementptr float, float* %305, i64 %2784
  %2786 = load float, float* %2785, align 4, !noalias !133
  %2787 = fmul float %2786, %2545
  %2788 = getelementptr float, float* %255, i64 %2784
  store float %2787, float* %2788, align 4, !noalias !133
  %2789 = add nuw nsw i64 %2538, 140
  %2790 = getelementptr float, float* %305, i64 %2789
  %2791 = getelementptr float, float* %255, i64 %2789
  %2792 = bitcast float* %2790 to <8 x float>*
  %2793 = load <8 x float>, <8 x float>* %2792, align 4, !noalias !133
  %2794 = fmul <8 x float> %2793, %2549
  %2795 = bitcast float* %2791 to <8 x float>*
  store <8 x float> %2794, <8 x float>* %2795, align 4, !noalias !133
  %2796 = add nuw nsw i64 %2538, 148
  %2797 = getelementptr float, float* %305, i64 %2796
  %2798 = getelementptr float, float* %255, i64 %2796
  %2799 = bitcast float* %2797 to <4 x float>*
  %2800 = load <4 x float>, <4 x float>* %2799, align 4, !noalias !133
  %2801 = fmul <4 x float> %2800, %2551
  %2802 = bitcast float* %2798 to <4 x float>*
  store <4 x float> %2801, <4 x float>* %2802, align 4, !noalias !133
  %2803 = add nuw nsw i64 %2538, 152
  %2804 = getelementptr float, float* %305, i64 %2803
  %2805 = load float, float* %2804, align 4, !noalias !133
  %2806 = fmul float %2805, %2545
  %2807 = getelementptr float, float* %255, i64 %2803
  store float %2806, float* %2807, align 4, !noalias !133
  %2808 = or i64 %2803, 1
  %2809 = getelementptr float, float* %305, i64 %2808
  %2810 = load float, float* %2809, align 4, !noalias !133
  %2811 = fmul float %2810, %2545
  %2812 = getelementptr float, float* %255, i64 %2808
  store float %2811, float* %2812, align 4, !noalias !133
  %2813 = add nuw nsw i64 %2538, 154
  %2814 = getelementptr float, float* %305, i64 %2813
  %2815 = getelementptr float, float* %255, i64 %2813
  %2816 = bitcast float* %2814 to <8 x float>*
  %2817 = load <8 x float>, <8 x float>* %2816, align 4, !noalias !133
  %2818 = fmul <8 x float> %2817, %2549
  %2819 = bitcast float* %2815 to <8 x float>*
  store <8 x float> %2818, <8 x float>* %2819, align 4, !noalias !133
  %2820 = add nuw nsw i64 %2538, 162
  %2821 = getelementptr float, float* %305, i64 %2820
  %2822 = getelementptr float, float* %255, i64 %2820
  %2823 = bitcast float* %2821 to <4 x float>*
  %2824 = load <4 x float>, <4 x float>* %2823, align 4, !noalias !133
  %2825 = fmul <4 x float> %2824, %2551
  %2826 = bitcast float* %2822 to <4 x float>*
  store <4 x float> %2825, <4 x float>* %2826, align 4, !noalias !133
  %2827 = add nuw nsw i64 %2538, 166
  %2828 = getelementptr float, float* %305, i64 %2827
  %2829 = load float, float* %2828, align 4, !noalias !133
  %2830 = fmul float %2829, %2545
  %2831 = getelementptr float, float* %255, i64 %2827
  store float %2830, float* %2831, align 4, !noalias !133
  %2832 = or i64 %2827, 1
  %2833 = getelementptr float, float* %305, i64 %2832
  %2834 = load float, float* %2833, align 4, !noalias !133
  %2835 = fmul float %2834, %2545
  %2836 = getelementptr float, float* %255, i64 %2832
  store float %2835, float* %2836, align 4, !noalias !133
  %2837 = add nuw nsw i64 %2538, 168
  %2838 = getelementptr float, float* %305, i64 %2837
  %2839 = getelementptr float, float* %255, i64 %2837
  %2840 = bitcast float* %2838 to <8 x float>*
  %2841 = load <8 x float>, <8 x float>* %2840, align 4, !noalias !133
  %2842 = fmul <8 x float> %2841, %2549
  %2843 = bitcast float* %2839 to <8 x float>*
  store <8 x float> %2842, <8 x float>* %2843, align 4, !noalias !133
  %2844 = add nuw nsw i64 %2538, 176
  %2845 = getelementptr float, float* %305, i64 %2844
  %2846 = getelementptr float, float* %255, i64 %2844
  %2847 = bitcast float* %2845 to <4 x float>*
  %2848 = load <4 x float>, <4 x float>* %2847, align 4, !noalias !133
  %2849 = fmul <4 x float> %2848, %2551
  %2850 = bitcast float* %2846 to <4 x float>*
  store <4 x float> %2849, <4 x float>* %2850, align 4, !noalias !133
  %2851 = add nuw nsw i64 %2538, 180
  %2852 = getelementptr float, float* %305, i64 %2851
  %2853 = load float, float* %2852, align 4, !noalias !133
  %2854 = fmul float %2853, %2545
  %2855 = getelementptr float, float* %255, i64 %2851
  store float %2854, float* %2855, align 4, !noalias !133
  %2856 = or i64 %2851, 1
  %2857 = getelementptr float, float* %305, i64 %2856
  %2858 = load float, float* %2857, align 4, !noalias !133
  %2859 = fmul float %2858, %2545
  %2860 = getelementptr float, float* %255, i64 %2856
  store float %2859, float* %2860, align 4, !noalias !133
  %2861 = add nuw nsw i64 %2538, 182
  %2862 = getelementptr float, float* %305, i64 %2861
  %2863 = getelementptr float, float* %255, i64 %2861
  %2864 = bitcast float* %2862 to <8 x float>*
  %2865 = load <8 x float>, <8 x float>* %2864, align 4, !noalias !133
  %2866 = fmul <8 x float> %2865, %2549
  %2867 = bitcast float* %2863 to <8 x float>*
  store <8 x float> %2866, <8 x float>* %2867, align 4, !noalias !133
  %2868 = add nuw nsw i64 %2538, 190
  %2869 = getelementptr float, float* %305, i64 %2868
  %2870 = getelementptr float, float* %255, i64 %2868
  %2871 = bitcast float* %2869 to <4 x float>*
  %2872 = load <4 x float>, <4 x float>* %2871, align 4, !noalias !133
  %2873 = fmul <4 x float> %2872, %2551
  %2874 = bitcast float* %2870 to <4 x float>*
  store <4 x float> %2873, <4 x float>* %2874, align 4, !noalias !133
  %2875 = add nuw nsw i64 %2538, 194
  %2876 = getelementptr float, float* %305, i64 %2875
  %2877 = load float, float* %2876, align 4, !noalias !133
  %2878 = fmul float %2877, %2545
  %2879 = getelementptr float, float* %255, i64 %2875
  store float %2878, float* %2879, align 4, !noalias !133
  %2880 = or i64 %2875, 1
  %2881 = getelementptr float, float* %305, i64 %2880
  %2882 = load float, float* %2881, align 4, !noalias !133
  %2883 = fmul float %2882, %2545
  %2884 = getelementptr float, float* %255, i64 %2880
  store float %2883, float* %2884, align 4, !noalias !133
  %2885 = add nuw nsw i64 %2537, 1
  %exitcond478.not.i = icmp eq i64 %2885, 480
  br i1 %exitcond478.not.i, label %exit407.i, label %cond408.preheader.i

exit407.i:                                        ; preds = %cond408.preheader.i
  %2886 = alloca [3 x i8*], align 8
  %2887 = alloca [3 x i64], align 16
  %2888 = alloca [8 x i64], align 8
  %2889 = alloca [3 x i8], align 1
  %.sub232.i = getelementptr inbounds [3 x i8], [3 x i8]* %2889, i64 0, i64 0
  %.sub231.i = getelementptr inbounds [8 x i64], [8 x i64]* %2888, i64 0, i64 0
  %.sub230.i = getelementptr inbounds [3 x i64], [3 x i64]* %2887, i64 0, i64 0
  %.sub.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %2886, i64 0, i64 0
  store i8* %malloccall82.i, i8** %.sub.i, align 8, !noalias !133
  store i8 6, i8* %.sub232.i, align 1, !noalias !133
  %2890 = bitcast [8 x i64]* %2888 to <4 x i64>*
  store <4 x i64> <i64 1, i64 112, i64 14, i64 14>, <4 x i64>* %2890, align 8, !noalias !133
  %2891 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2886, i64 0, i64 1
  store i8* %malloccall14.i, i8** %2891, align 8, !noalias !133
  %2892 = getelementptr inbounds [3 x i8], [3 x i8]* %2889, i64 0, i64 1
  store i8 6, i8* %2892, align 1, !noalias !133
  %2893 = bitcast [3 x i64]* %2887 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %2893, align 16, !noalias !133
  %2894 = getelementptr inbounds [8 x i64], [8 x i64]* %2888, i64 0, i64 4
  %2895 = bitcast i64* %2894 to <4 x i64>*
  store <4 x i64> <i64 1, i64 480, i64 14, i64 14>, <4 x i64>* %2895, align 8, !noalias !133
  %2896 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2886, i64 0, i64 2
  %2897 = bitcast i8** %2896 to float**
  store float* %143, float** %2897, align 8, !noalias !133
  %2898 = getelementptr inbounds [3 x i8], [3 x i8]* %2889, i64 0, i64 2
  store i8 6, i8* %2898, align 1, !noalias !133
  %2899 = getelementptr inbounds [3 x i64], [3 x i64]* %2887, i64 0, i64 2
  store i64 0, i64* %2899, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub.i, i64* nonnull %.sub230.i, i64* nonnull %.sub231.i, i8* nonnull %.sub232.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %2900 = alloca [3 x i8*], align 8
  %2901 = alloca [3 x i64], align 16
  %2902 = alloca [8 x i64], align 8
  %2903 = alloca [3 x i8], align 1
  %.sub237.i = getelementptr inbounds [3 x i8], [3 x i8]* %2903, i64 0, i64 0
  %.sub236.i = getelementptr inbounds [8 x i64], [8 x i64]* %2902, i64 0, i64 0
  %.sub235.i = getelementptr inbounds [3 x i64], [3 x i64]* %2901, i64 0, i64 0
  %.sub234.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %2900, i64 0, i64 0
  store i8* %malloccall13.i, i8** %.sub234.i, align 8, !noalias !133
  store i8 6, i8* %.sub237.i, align 1, !noalias !133
  %2904 = bitcast [8 x i64]* %2902 to <4 x i64>*
  store <4 x i64> <i64 1, i64 672, i64 14, i64 14>, <4 x i64>* %2904, align 8, !noalias !133
  %2905 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2900, i64 0, i64 1
  store i8* %malloccall82.i, i8** %2905, align 8, !noalias !133
  %2906 = getelementptr inbounds [3 x i8], [3 x i8]* %2903, i64 0, i64 1
  store i8 6, i8* %2906, align 1, !noalias !133
  %2907 = bitcast [3 x i64]* %2901 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %2907, align 16, !noalias !133
  %2908 = getelementptr inbounds [8 x i64], [8 x i64]* %2902, i64 0, i64 4
  %2909 = bitcast i64* %2908 to <4 x i64>*
  store <4 x i64> <i64 1, i64 112, i64 14, i64 14>, <4 x i64>* %2909, align 8, !noalias !133
  %2910 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2900, i64 0, i64 2
  %2911 = bitcast i8** %2910 to float**
  store float* %146, float** %2911, align 8, !noalias !133
  %2912 = getelementptr inbounds [3 x i8], [3 x i8]* %2903, i64 0, i64 2
  store i8 6, i8* %2912, align 1, !noalias !133
  %2913 = getelementptr inbounds [3 x i64], [3 x i64]* %2901, i64 0, i64 2
  store i64 0, i64* %2913, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub234.i, i64* nonnull %.sub235.i, i64* nonnull %.sub236.i, i8* nonnull %.sub237.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond423.preheader.i

cond423.preheader.i:                              ; preds = %exit425.i, %exit407.i
  %2914 = phi i64 [ 0, %exit407.i ], [ %2983, %exit425.i ]
  %2915 = mul nuw nsw i64 %2914, 196
  %2916 = add nuw nsw i64 %2915, 8
  %2917 = add nuw nsw i64 %2915, 12
  br label %cond426.preheader.i

exit422.i:                                        ; preds = %exit425.i
  %2918 = alloca [3 x i8*], align 8
  %2919 = alloca [3 x i64], align 16
  %2920 = alloca [8 x i64], align 8
  %2921 = alloca [3 x i8], align 1
  %.sub242.i = getelementptr inbounds [3 x i8], [3 x i8]* %2921, i64 0, i64 0
  %.sub241.i = getelementptr inbounds [8 x i64], [8 x i64]* %2920, i64 0, i64 0
  %.sub240.i = getelementptr inbounds [3 x i64], [3 x i64]* %2919, i64 0, i64 0
  %.sub239.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %2918, i64 0, i64 0
  store i8* %malloccall12.i, i8** %.sub239.i, align 8, !noalias !133
  store i8 6, i8* %.sub242.i, align 1, !noalias !133
  %2922 = bitcast [8 x i64]* %2920 to <4 x i64>*
  store <4 x i64> <i64 1, i64 672, i64 14, i64 14>, <4 x i64>* %2922, align 8, !noalias !133
  %2923 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2918, i64 0, i64 1
  store i8* %malloccall33.i, i8** %2923, align 8, !noalias !133
  %2924 = getelementptr inbounds [3 x i8], [3 x i8]* %2921, i64 0, i64 1
  store i8 6, i8* %2924, align 1, !noalias !133
  %2925 = bitcast [3 x i64]* %2919 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %2925, align 16, !noalias !133
  %2926 = getelementptr inbounds [8 x i64], [8 x i64]* %2920, i64 0, i64 4
  %2927 = bitcast i64* %2926 to <4 x i64>*
  store <4 x i64> <i64 1, i64 672, i64 14, i64 14>, <4 x i64>* %2927, align 8, !noalias !133
  %2928 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2918, i64 0, i64 2
  %2929 = bitcast i8** %2928 to float**
  store float* %149, float** %2929, align 8, !noalias !133
  %2930 = getelementptr inbounds [3 x i8], [3 x i8]* %2921, i64 0, i64 2
  store i8 6, i8* %2930, align 1, !noalias !133
  %2931 = getelementptr inbounds [3 x i64], [3 x i64]* %2919, i64 0, i64 2
  store i64 0, i64* %2931, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub239.i, i64* nonnull %.sub240.i, i64* nonnull %.sub241.i, i8* nonnull %.sub242.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond438.preheader.i

cond426.preheader.i:                              ; preds = %cond426.preheader.i, %cond423.preheader.i
  %2932 = phi i64 [ 0, %cond423.preheader.i ], [ %2982, %cond426.preheader.i ]
  %2933 = mul nuw nsw i64 %2932, 14
  %2934 = add nuw nsw i64 %2933, %2915
  %2935 = getelementptr float, float* %254, i64 %2934
  %2936 = getelementptr float, float* %273, i64 %2934
  %2937 = bitcast float* %2935 to <8 x float>*
  %2938 = load <8 x float>, <8 x float>* %2937, align 4, !noalias !133
  %2939 = fadd <8 x float> %2938, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %2940 = fcmp olt <8 x float> %2939, zeroinitializer
  %2941 = select <8 x i1> %2940, <8 x float> zeroinitializer, <8 x float> %2939
  %2942 = fcmp ogt <8 x float> %2941, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2943 = select <8 x i1> %2942, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %2941
  %2944 = fmul <8 x float> %2938, %2943
  %2945 = fdiv <8 x float> %2944, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2946 = bitcast float* %2936 to <8 x float>*
  store <8 x float> %2945, <8 x float>* %2946, align 4, !noalias !133
  %2947 = add nuw nsw i64 %2916, %2933
  %2948 = getelementptr float, float* %254, i64 %2947
  %2949 = getelementptr float, float* %273, i64 %2947
  %2950 = bitcast float* %2948 to <4 x float>*
  %2951 = load <4 x float>, <4 x float>* %2950, align 4, !noalias !133
  %2952 = fadd <4 x float> %2951, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %2953 = fcmp olt <4 x float> %2952, zeroinitializer
  %2954 = select <4 x i1> %2953, <4 x float> zeroinitializer, <4 x float> %2952
  %2955 = fcmp ogt <4 x float> %2954, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2956 = select <4 x i1> %2955, <4 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <4 x float> %2954
  %2957 = fmul <4 x float> %2951, %2956
  %2958 = fdiv <4 x float> %2957, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %2959 = bitcast float* %2949 to <4 x float>*
  store <4 x float> %2958, <4 x float>* %2959, align 4, !noalias !133
  %2960 = add nuw nsw i64 %2917, %2933
  %2961 = getelementptr float, float* %254, i64 %2960
  %2962 = load float, float* %2961, align 4, !noalias !133
  %2963 = fadd float %2962, 3.000000e+00
  %2964 = fcmp olt float %2963, 0.000000e+00
  %2965 = select i1 %2964, float 0.000000e+00, float %2963
  %2966 = fcmp ogt float %2965, 6.000000e+00
  %2967 = select i1 %2966, float 6.000000e+00, float %2965
  %2968 = fmul float %2962, %2967
  %2969 = fdiv float %2968, 6.000000e+00
  %2970 = getelementptr float, float* %273, i64 %2960
  store float %2969, float* %2970, align 4, !noalias !133
  %2971 = or i64 %2960, 1
  %2972 = getelementptr float, float* %254, i64 %2971
  %2973 = load float, float* %2972, align 4, !noalias !133
  %2974 = fadd float %2973, 3.000000e+00
  %2975 = fcmp olt float %2974, 0.000000e+00
  %2976 = select i1 %2975, float 0.000000e+00, float %2974
  %2977 = fcmp ogt float %2976, 6.000000e+00
  %2978 = select i1 %2977, float 6.000000e+00, float %2976
  %2979 = fmul float %2973, %2978
  %2980 = fdiv float %2979, 6.000000e+00
  %2981 = getelementptr float, float* %273, i64 %2971
  store float %2980, float* %2981, align 4, !noalias !133
  %2982 = add nuw nsw i64 %2932, 1
  %exitcond473.not.i = icmp eq i64 %2982, 14
  br i1 %exitcond473.not.i, label %exit425.i, label %cond426.preheader.i

exit425.i:                                        ; preds = %cond426.preheader.i
  %2983 = add nuw nsw i64 %2914, 1
  %exitcond474.not.i = icmp eq i64 %2983, 672
  br i1 %exitcond474.not.i, label %exit422.i, label %cond423.preheader.i

cond438.preheader.i:                              ; preds = %exit440.i, %exit422.i
  %2984 = phi i64 [ 0, %exit422.i ], [ %3077, %exit440.i ]
  %2985 = mul nuw nsw i64 %2984, 196
  %2986 = add nuw nsw i64 %2985, 8
  %2987 = add nuw nsw i64 %2985, 12
  br label %cond441.preheader.i

exit437.i:                                        ; preds = %exit440.i
  %2988 = alloca [2 x i8*], align 8
  %2989 = alloca <2 x i64>, align 16
  %2990 = alloca [8 x i64], align 8
  %2991 = alloca [2 x i8], align 1
  %2992 = alloca <2 x i64>, align 16
  %.sub248.i = getelementptr inbounds <2 x i64>, <2 x i64>* %2992, i64 0, i64 0
  %.sub247.i = getelementptr inbounds [2 x i8], [2 x i8]* %2991, i64 0, i64 0
  %.sub246.i = getelementptr inbounds [8 x i64], [8 x i64]* %2990, i64 0, i64 0
  %.sub245.i = getelementptr inbounds <2 x i64>, <2 x i64>* %2989, i64 0, i64 0
  %.sub244.i = getelementptr inbounds [2 x i8*], [2 x i8*]* %2988, i64 0, i64 0
  store i8* %malloccall59.i, i8** %.sub244.i, align 8, !noalias !133
  store i8 6, i8* %.sub247.i, align 1, !noalias !133
  %2993 = bitcast [8 x i64]* %2990 to <4 x i64>*
  store <4 x i64> <i64 1, i64 672, i64 1, i64 1>, <4 x i64>* %2993, align 8, !noalias !133
  %2994 = getelementptr inbounds [2 x i8*], [2 x i8*]* %2988, i64 0, i64 1
  store i8* %malloccall25.i, i8** %2994, align 8, !noalias !133
  %2995 = getelementptr inbounds [2 x i8], [2 x i8]* %2991, i64 0, i64 1
  store i8 6, i8* %2995, align 1, !noalias !133
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %2989, align 16, !noalias !133
  %2996 = getelementptr inbounds [8 x i64], [8 x i64]* %2990, i64 0, i64 4
  %2997 = bitcast i64* %2996 to <4 x i64>*
  store <4 x i64> <i64 1, i64 672, i64 14, i64 14>, <4 x i64>* %2997, align 8, !noalias !133
  store <2 x i64> <i64 1, i64 1>, <2 x i64>* %2992, align 16, !noalias !133
  call void @nnc_aten_adaptive_avg_pool2d(i64 2, i8** nonnull %.sub244.i, i64* nonnull %.sub245.i, i64* nonnull %.sub246.i, i8* nonnull %.sub247.i, i64 2, i64* nonnull %.sub248.i) #0, !noalias !135
  %2998 = alloca [3 x i8*], align 8
  %2999 = alloca [3 x i64], align 16
  %3000 = alloca [8 x i64], align 8
  %3001 = alloca [3 x i8], align 1
  %.sub252.i = getelementptr inbounds [3 x i8], [3 x i8]* %3001, i64 0, i64 0
  %.sub251.i = getelementptr inbounds [8 x i64], [8 x i64]* %3000, i64 0, i64 0
  %.sub250.i = getelementptr inbounds [3 x i64], [3 x i64]* %2999, i64 0, i64 0
  %.sub249.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %2998, i64 0, i64 0
  store i8* %malloccall11.i, i8** %.sub249.i, align 8, !noalias !133
  store i8 6, i8* %.sub252.i, align 1, !noalias !133
  %3002 = bitcast [8 x i64]* %3000 to <4 x i64>*
  store <4 x i64> <i64 1, i64 168, i64 1, i64 1>, <4 x i64>* %3002, align 8, !noalias !133
  %3003 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2998, i64 0, i64 1
  store i8* %malloccall59.i, i8** %3003, align 8, !noalias !133
  %3004 = getelementptr inbounds [3 x i8], [3 x i8]* %3001, i64 0, i64 1
  store i8 6, i8* %3004, align 1, !noalias !133
  %3005 = bitcast [3 x i64]* %2999 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %3005, align 16, !noalias !133
  %3006 = getelementptr inbounds [8 x i64], [8 x i64]* %3000, i64 0, i64 4
  %3007 = bitcast i64* %3006 to <4 x i64>*
  store <4 x i64> <i64 1, i64 672, i64 1, i64 1>, <4 x i64>* %3007, align 8, !noalias !133
  %3008 = getelementptr inbounds [3 x i8*], [3 x i8*]* %2998, i64 0, i64 2
  %3009 = bitcast i8** %3008 to float**
  store float* %152, float** %3009, align 8, !noalias !133
  %3010 = getelementptr inbounds [3 x i8], [3 x i8]* %3001, i64 0, i64 2
  store i8 6, i8* %3010, align 1, !noalias !133
  %3011 = getelementptr inbounds [3 x i64], [3 x i64]* %2999, i64 0, i64 2
  store i64 0, i64* %3011, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub249.i, i64* nonnull %.sub250.i, i64* nonnull %.sub251.i, i8* nonnull %.sub252.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %3012 = alloca [3 x i8*], align 8
  %3013 = alloca [3 x i64], align 16
  %3014 = alloca [8 x i64], align 8
  %3015 = alloca [3 x i8], align 1
  %.sub257.i = getelementptr inbounds [3 x i8], [3 x i8]* %3015, i64 0, i64 0
  %.sub256.i = getelementptr inbounds [8 x i64], [8 x i64]* %3014, i64 0, i64 0
  %.sub255.i = getelementptr inbounds [3 x i64], [3 x i64]* %3013, i64 0, i64 0
  %.sub254.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %3012, i64 0, i64 0
  store i8* %malloccall10.i, i8** %.sub254.i, align 8, !noalias !133
  store i8 6, i8* %.sub257.i, align 1, !noalias !133
  %3016 = bitcast [8 x i64]* %3014 to <4 x i64>*
  store <4 x i64> <i64 1, i64 672, i64 1, i64 1>, <4 x i64>* %3016, align 8, !noalias !133
  %3017 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3012, i64 0, i64 1
  store i8* %malloccall11.i, i8** %3017, align 8, !noalias !133
  %3018 = getelementptr inbounds [3 x i8], [3 x i8]* %3015, i64 0, i64 1
  store i8 6, i8* %3018, align 1, !noalias !133
  %3019 = bitcast [3 x i64]* %3013 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %3019, align 16, !noalias !133
  %3020 = getelementptr inbounds [8 x i64], [8 x i64]* %3014, i64 0, i64 4
  %3021 = bitcast i64* %3020 to <4 x i64>*
  store <4 x i64> <i64 1, i64 168, i64 1, i64 1>, <4 x i64>* %3021, align 8, !noalias !133
  %3022 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3012, i64 0, i64 2
  %3023 = bitcast i8** %3022 to float**
  store float* %155, float** %3023, align 8, !noalias !133
  %3024 = getelementptr inbounds [3 x i8], [3 x i8]* %3015, i64 0, i64 2
  store i8 6, i8* %3024, align 1, !noalias !133
  %3025 = getelementptr inbounds [3 x i64], [3 x i64]* %3013, i64 0, i64 2
  store i64 0, i64* %3025, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub254.i, i64* nonnull %.sub255.i, i64* nonnull %.sub256.i, i8* nonnull %.sub257.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond453.preheader.i

cond441.preheader.i:                              ; preds = %cond441.preheader.i, %cond438.preheader.i
  %3026 = phi i64 [ 0, %cond438.preheader.i ], [ %3076, %cond441.preheader.i ]
  %3027 = mul nuw nsw i64 %3026, 14
  %3028 = add nuw nsw i64 %3027, %2985
  %3029 = getelementptr float, float* %253, i64 %3028
  %3030 = getelementptr float, float* %265, i64 %3028
  %3031 = bitcast float* %3029 to <8 x float>*
  %3032 = load <8 x float>, <8 x float>* %3031, align 4, !noalias !133
  %3033 = fadd <8 x float> %3032, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %3034 = fcmp olt <8 x float> %3033, zeroinitializer
  %3035 = select <8 x i1> %3034, <8 x float> zeroinitializer, <8 x float> %3033
  %3036 = fcmp ogt <8 x float> %3035, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %3037 = select <8 x i1> %3036, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %3035
  %3038 = fmul <8 x float> %3032, %3037
  %3039 = fdiv <8 x float> %3038, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %3040 = bitcast float* %3030 to <8 x float>*
  store <8 x float> %3039, <8 x float>* %3040, align 4, !noalias !133
  %3041 = add nuw nsw i64 %2986, %3027
  %3042 = getelementptr float, float* %253, i64 %3041
  %3043 = getelementptr float, float* %265, i64 %3041
  %3044 = bitcast float* %3042 to <4 x float>*
  %3045 = load <4 x float>, <4 x float>* %3044, align 4, !noalias !133
  %3046 = fadd <4 x float> %3045, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %3047 = fcmp olt <4 x float> %3046, zeroinitializer
  %3048 = select <4 x i1> %3047, <4 x float> zeroinitializer, <4 x float> %3046
  %3049 = fcmp ogt <4 x float> %3048, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %3050 = select <4 x i1> %3049, <4 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <4 x float> %3048
  %3051 = fmul <4 x float> %3045, %3050
  %3052 = fdiv <4 x float> %3051, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %3053 = bitcast float* %3043 to <4 x float>*
  store <4 x float> %3052, <4 x float>* %3053, align 4, !noalias !133
  %3054 = add nuw nsw i64 %2987, %3027
  %3055 = getelementptr float, float* %253, i64 %3054
  %3056 = load float, float* %3055, align 4, !noalias !133
  %3057 = fadd float %3056, 3.000000e+00
  %3058 = fcmp olt float %3057, 0.000000e+00
  %3059 = select i1 %3058, float 0.000000e+00, float %3057
  %3060 = fcmp ogt float %3059, 6.000000e+00
  %3061 = select i1 %3060, float 6.000000e+00, float %3059
  %3062 = fmul float %3056, %3061
  %3063 = fdiv float %3062, 6.000000e+00
  %3064 = getelementptr float, float* %265, i64 %3054
  store float %3063, float* %3064, align 4, !noalias !133
  %3065 = or i64 %3054, 1
  %3066 = getelementptr float, float* %253, i64 %3065
  %3067 = load float, float* %3066, align 4, !noalias !133
  %3068 = fadd float %3067, 3.000000e+00
  %3069 = fcmp olt float %3068, 0.000000e+00
  %3070 = select i1 %3069, float 0.000000e+00, float %3068
  %3071 = fcmp ogt float %3070, 6.000000e+00
  %3072 = select i1 %3071, float 6.000000e+00, float %3070
  %3073 = fmul float %3067, %3072
  %3074 = fdiv float %3073, 6.000000e+00
  %3075 = getelementptr float, float* %265, i64 %3065
  store float %3074, float* %3075, align 4, !noalias !133
  %3076 = add nuw nsw i64 %3026, 1
  %exitcond469.not.i = icmp eq i64 %3076, 14
  br i1 %exitcond469.not.i, label %exit440.i, label %cond441.preheader.i

exit440.i:                                        ; preds = %cond441.preheader.i
  %3077 = add nuw nsw i64 %2984, 1
  %exitcond470.not.i = icmp eq i64 %3077, 672
  br i1 %exitcond470.not.i, label %exit437.i, label %cond438.preheader.i

cond453.preheader.i:                              ; preds = %cond453.preheader.i, %exit437.i
  %3078 = phi i64 [ 0, %exit437.i ], [ %3426, %cond453.preheader.i ]
  %3079 = mul nuw nsw i64 %3078, 196
  %3080 = getelementptr float, float* %252, i64 %3078
  %3081 = load float, float* %3080, align 4, !noalias !133
  %3082 = fadd float %3081, 3.000000e+00
  %3083 = fcmp olt float %3082, 0.000000e+00
  %3084 = select i1 %3083, float 0.000000e+00, float %3082
  %3085 = fcmp ogt float %3084, 6.000000e+00
  %.op390.i = fdiv float %3082, 6.000000e+00
  %.op389.i = select i1 %3083, float 0.000000e+00, float %.op390.i
  %3086 = select i1 %3085, float 1.000000e+00, float %.op389.i
  %3087 = add nuw nsw i64 %3079, 8
  %3088 = add nuw nsw i64 %3079, 12
  %3089 = insertelement <8 x float> poison, float %3086, i32 0
  %3090 = shufflevector <8 x float> %3089, <8 x float> undef, <8 x i32> zeroinitializer
  %3091 = insertelement <4 x float> poison, float %3086, i32 0
  %3092 = shufflevector <4 x float> %3091, <4 x float> undef, <4 x i32> zeroinitializer
  %3093 = getelementptr float, float* %265, i64 %3079
  %3094 = getelementptr float, float* %276, i64 %3079
  %3095 = bitcast float* %3093 to <8 x float>*
  %3096 = load <8 x float>, <8 x float>* %3095, align 4, !noalias !133
  %3097 = fmul <8 x float> %3096, %3090
  %3098 = bitcast float* %3094 to <8 x float>*
  store <8 x float> %3097, <8 x float>* %3098, align 4, !noalias !133
  %3099 = getelementptr float, float* %265, i64 %3087
  %3100 = getelementptr float, float* %276, i64 %3087
  %3101 = bitcast float* %3099 to <4 x float>*
  %3102 = load <4 x float>, <4 x float>* %3101, align 4, !noalias !133
  %3103 = fmul <4 x float> %3102, %3092
  %3104 = bitcast float* %3100 to <4 x float>*
  store <4 x float> %3103, <4 x float>* %3104, align 4, !noalias !133
  %3105 = getelementptr float, float* %265, i64 %3088
  %3106 = load float, float* %3105, align 4, !noalias !133
  %3107 = fmul float %3106, %3086
  %3108 = getelementptr float, float* %276, i64 %3088
  store float %3107, float* %3108, align 4, !noalias !133
  %3109 = or i64 %3088, 1
  %3110 = getelementptr float, float* %265, i64 %3109
  %3111 = load float, float* %3110, align 4, !noalias !133
  %3112 = fmul float %3111, %3086
  %3113 = getelementptr float, float* %276, i64 %3109
  store float %3112, float* %3113, align 4, !noalias !133
  %3114 = add nuw nsw i64 %3079, 14
  %3115 = getelementptr float, float* %265, i64 %3114
  %3116 = getelementptr float, float* %276, i64 %3114
  %3117 = bitcast float* %3115 to <8 x float>*
  %3118 = load <8 x float>, <8 x float>* %3117, align 4, !noalias !133
  %3119 = fmul <8 x float> %3118, %3090
  %3120 = bitcast float* %3116 to <8 x float>*
  store <8 x float> %3119, <8 x float>* %3120, align 4, !noalias !133
  %3121 = add nuw nsw i64 %3079, 22
  %3122 = getelementptr float, float* %265, i64 %3121
  %3123 = getelementptr float, float* %276, i64 %3121
  %3124 = bitcast float* %3122 to <4 x float>*
  %3125 = load <4 x float>, <4 x float>* %3124, align 4, !noalias !133
  %3126 = fmul <4 x float> %3125, %3092
  %3127 = bitcast float* %3123 to <4 x float>*
  store <4 x float> %3126, <4 x float>* %3127, align 4, !noalias !133
  %3128 = add nuw nsw i64 %3079, 26
  %3129 = getelementptr float, float* %265, i64 %3128
  %3130 = load float, float* %3129, align 4, !noalias !133
  %3131 = fmul float %3130, %3086
  %3132 = getelementptr float, float* %276, i64 %3128
  store float %3131, float* %3132, align 4, !noalias !133
  %3133 = or i64 %3128, 1
  %3134 = getelementptr float, float* %265, i64 %3133
  %3135 = load float, float* %3134, align 4, !noalias !133
  %3136 = fmul float %3135, %3086
  %3137 = getelementptr float, float* %276, i64 %3133
  store float %3136, float* %3137, align 4, !noalias !133
  %3138 = add nuw nsw i64 %3079, 28
  %3139 = getelementptr float, float* %265, i64 %3138
  %3140 = getelementptr float, float* %276, i64 %3138
  %3141 = bitcast float* %3139 to <8 x float>*
  %3142 = load <8 x float>, <8 x float>* %3141, align 4, !noalias !133
  %3143 = fmul <8 x float> %3142, %3090
  %3144 = bitcast float* %3140 to <8 x float>*
  store <8 x float> %3143, <8 x float>* %3144, align 4, !noalias !133
  %3145 = add nuw nsw i64 %3079, 36
  %3146 = getelementptr float, float* %265, i64 %3145
  %3147 = getelementptr float, float* %276, i64 %3145
  %3148 = bitcast float* %3146 to <4 x float>*
  %3149 = load <4 x float>, <4 x float>* %3148, align 4, !noalias !133
  %3150 = fmul <4 x float> %3149, %3092
  %3151 = bitcast float* %3147 to <4 x float>*
  store <4 x float> %3150, <4 x float>* %3151, align 4, !noalias !133
  %3152 = add nuw nsw i64 %3079, 40
  %3153 = getelementptr float, float* %265, i64 %3152
  %3154 = load float, float* %3153, align 4, !noalias !133
  %3155 = fmul float %3154, %3086
  %3156 = getelementptr float, float* %276, i64 %3152
  store float %3155, float* %3156, align 4, !noalias !133
  %3157 = or i64 %3152, 1
  %3158 = getelementptr float, float* %265, i64 %3157
  %3159 = load float, float* %3158, align 4, !noalias !133
  %3160 = fmul float %3159, %3086
  %3161 = getelementptr float, float* %276, i64 %3157
  store float %3160, float* %3161, align 4, !noalias !133
  %3162 = add nuw nsw i64 %3079, 42
  %3163 = getelementptr float, float* %265, i64 %3162
  %3164 = getelementptr float, float* %276, i64 %3162
  %3165 = bitcast float* %3163 to <8 x float>*
  %3166 = load <8 x float>, <8 x float>* %3165, align 4, !noalias !133
  %3167 = fmul <8 x float> %3166, %3090
  %3168 = bitcast float* %3164 to <8 x float>*
  store <8 x float> %3167, <8 x float>* %3168, align 4, !noalias !133
  %3169 = add nuw nsw i64 %3079, 50
  %3170 = getelementptr float, float* %265, i64 %3169
  %3171 = getelementptr float, float* %276, i64 %3169
  %3172 = bitcast float* %3170 to <4 x float>*
  %3173 = load <4 x float>, <4 x float>* %3172, align 4, !noalias !133
  %3174 = fmul <4 x float> %3173, %3092
  %3175 = bitcast float* %3171 to <4 x float>*
  store <4 x float> %3174, <4 x float>* %3175, align 4, !noalias !133
  %3176 = add nuw nsw i64 %3079, 54
  %3177 = getelementptr float, float* %265, i64 %3176
  %3178 = load float, float* %3177, align 4, !noalias !133
  %3179 = fmul float %3178, %3086
  %3180 = getelementptr float, float* %276, i64 %3176
  store float %3179, float* %3180, align 4, !noalias !133
  %3181 = or i64 %3176, 1
  %3182 = getelementptr float, float* %265, i64 %3181
  %3183 = load float, float* %3182, align 4, !noalias !133
  %3184 = fmul float %3183, %3086
  %3185 = getelementptr float, float* %276, i64 %3181
  store float %3184, float* %3185, align 4, !noalias !133
  %3186 = add nuw nsw i64 %3079, 56
  %3187 = getelementptr float, float* %265, i64 %3186
  %3188 = getelementptr float, float* %276, i64 %3186
  %3189 = bitcast float* %3187 to <8 x float>*
  %3190 = load <8 x float>, <8 x float>* %3189, align 4, !noalias !133
  %3191 = fmul <8 x float> %3190, %3090
  %3192 = bitcast float* %3188 to <8 x float>*
  store <8 x float> %3191, <8 x float>* %3192, align 4, !noalias !133
  %3193 = add nuw nsw i64 %3079, 64
  %3194 = getelementptr float, float* %265, i64 %3193
  %3195 = getelementptr float, float* %276, i64 %3193
  %3196 = bitcast float* %3194 to <4 x float>*
  %3197 = load <4 x float>, <4 x float>* %3196, align 4, !noalias !133
  %3198 = fmul <4 x float> %3197, %3092
  %3199 = bitcast float* %3195 to <4 x float>*
  store <4 x float> %3198, <4 x float>* %3199, align 4, !noalias !133
  %3200 = add nuw nsw i64 %3079, 68
  %3201 = getelementptr float, float* %265, i64 %3200
  %3202 = load float, float* %3201, align 4, !noalias !133
  %3203 = fmul float %3202, %3086
  %3204 = getelementptr float, float* %276, i64 %3200
  store float %3203, float* %3204, align 4, !noalias !133
  %3205 = or i64 %3200, 1
  %3206 = getelementptr float, float* %265, i64 %3205
  %3207 = load float, float* %3206, align 4, !noalias !133
  %3208 = fmul float %3207, %3086
  %3209 = getelementptr float, float* %276, i64 %3205
  store float %3208, float* %3209, align 4, !noalias !133
  %3210 = add nuw nsw i64 %3079, 70
  %3211 = getelementptr float, float* %265, i64 %3210
  %3212 = getelementptr float, float* %276, i64 %3210
  %3213 = bitcast float* %3211 to <8 x float>*
  %3214 = load <8 x float>, <8 x float>* %3213, align 4, !noalias !133
  %3215 = fmul <8 x float> %3214, %3090
  %3216 = bitcast float* %3212 to <8 x float>*
  store <8 x float> %3215, <8 x float>* %3216, align 4, !noalias !133
  %3217 = add nuw nsw i64 %3079, 78
  %3218 = getelementptr float, float* %265, i64 %3217
  %3219 = getelementptr float, float* %276, i64 %3217
  %3220 = bitcast float* %3218 to <4 x float>*
  %3221 = load <4 x float>, <4 x float>* %3220, align 4, !noalias !133
  %3222 = fmul <4 x float> %3221, %3092
  %3223 = bitcast float* %3219 to <4 x float>*
  store <4 x float> %3222, <4 x float>* %3223, align 4, !noalias !133
  %3224 = add nuw nsw i64 %3079, 82
  %3225 = getelementptr float, float* %265, i64 %3224
  %3226 = load float, float* %3225, align 4, !noalias !133
  %3227 = fmul float %3226, %3086
  %3228 = getelementptr float, float* %276, i64 %3224
  store float %3227, float* %3228, align 4, !noalias !133
  %3229 = or i64 %3224, 1
  %3230 = getelementptr float, float* %265, i64 %3229
  %3231 = load float, float* %3230, align 4, !noalias !133
  %3232 = fmul float %3231, %3086
  %3233 = getelementptr float, float* %276, i64 %3229
  store float %3232, float* %3233, align 4, !noalias !133
  %3234 = add nuw nsw i64 %3079, 84
  %3235 = getelementptr float, float* %265, i64 %3234
  %3236 = getelementptr float, float* %276, i64 %3234
  %3237 = bitcast float* %3235 to <8 x float>*
  %3238 = load <8 x float>, <8 x float>* %3237, align 4, !noalias !133
  %3239 = fmul <8 x float> %3238, %3090
  %3240 = bitcast float* %3236 to <8 x float>*
  store <8 x float> %3239, <8 x float>* %3240, align 4, !noalias !133
  %3241 = add nuw nsw i64 %3079, 92
  %3242 = getelementptr float, float* %265, i64 %3241
  %3243 = getelementptr float, float* %276, i64 %3241
  %3244 = bitcast float* %3242 to <4 x float>*
  %3245 = load <4 x float>, <4 x float>* %3244, align 4, !noalias !133
  %3246 = fmul <4 x float> %3245, %3092
  %3247 = bitcast float* %3243 to <4 x float>*
  store <4 x float> %3246, <4 x float>* %3247, align 4, !noalias !133
  %3248 = add nuw nsw i64 %3079, 96
  %3249 = getelementptr float, float* %265, i64 %3248
  %3250 = load float, float* %3249, align 4, !noalias !133
  %3251 = fmul float %3250, %3086
  %3252 = getelementptr float, float* %276, i64 %3248
  store float %3251, float* %3252, align 4, !noalias !133
  %3253 = or i64 %3248, 1
  %3254 = getelementptr float, float* %265, i64 %3253
  %3255 = load float, float* %3254, align 4, !noalias !133
  %3256 = fmul float %3255, %3086
  %3257 = getelementptr float, float* %276, i64 %3253
  store float %3256, float* %3257, align 4, !noalias !133
  %3258 = add nuw nsw i64 %3079, 98
  %3259 = getelementptr float, float* %265, i64 %3258
  %3260 = getelementptr float, float* %276, i64 %3258
  %3261 = bitcast float* %3259 to <8 x float>*
  %3262 = load <8 x float>, <8 x float>* %3261, align 4, !noalias !133
  %3263 = fmul <8 x float> %3262, %3090
  %3264 = bitcast float* %3260 to <8 x float>*
  store <8 x float> %3263, <8 x float>* %3264, align 4, !noalias !133
  %3265 = add nuw nsw i64 %3079, 106
  %3266 = getelementptr float, float* %265, i64 %3265
  %3267 = getelementptr float, float* %276, i64 %3265
  %3268 = bitcast float* %3266 to <4 x float>*
  %3269 = load <4 x float>, <4 x float>* %3268, align 4, !noalias !133
  %3270 = fmul <4 x float> %3269, %3092
  %3271 = bitcast float* %3267 to <4 x float>*
  store <4 x float> %3270, <4 x float>* %3271, align 4, !noalias !133
  %3272 = add nuw nsw i64 %3079, 110
  %3273 = getelementptr float, float* %265, i64 %3272
  %3274 = load float, float* %3273, align 4, !noalias !133
  %3275 = fmul float %3274, %3086
  %3276 = getelementptr float, float* %276, i64 %3272
  store float %3275, float* %3276, align 4, !noalias !133
  %3277 = or i64 %3272, 1
  %3278 = getelementptr float, float* %265, i64 %3277
  %3279 = load float, float* %3278, align 4, !noalias !133
  %3280 = fmul float %3279, %3086
  %3281 = getelementptr float, float* %276, i64 %3277
  store float %3280, float* %3281, align 4, !noalias !133
  %3282 = add nuw nsw i64 %3079, 112
  %3283 = getelementptr float, float* %265, i64 %3282
  %3284 = getelementptr float, float* %276, i64 %3282
  %3285 = bitcast float* %3283 to <8 x float>*
  %3286 = load <8 x float>, <8 x float>* %3285, align 4, !noalias !133
  %3287 = fmul <8 x float> %3286, %3090
  %3288 = bitcast float* %3284 to <8 x float>*
  store <8 x float> %3287, <8 x float>* %3288, align 4, !noalias !133
  %3289 = add nuw nsw i64 %3079, 120
  %3290 = getelementptr float, float* %265, i64 %3289
  %3291 = getelementptr float, float* %276, i64 %3289
  %3292 = bitcast float* %3290 to <4 x float>*
  %3293 = load <4 x float>, <4 x float>* %3292, align 4, !noalias !133
  %3294 = fmul <4 x float> %3293, %3092
  %3295 = bitcast float* %3291 to <4 x float>*
  store <4 x float> %3294, <4 x float>* %3295, align 4, !noalias !133
  %3296 = add nuw nsw i64 %3079, 124
  %3297 = getelementptr float, float* %265, i64 %3296
  %3298 = load float, float* %3297, align 4, !noalias !133
  %3299 = fmul float %3298, %3086
  %3300 = getelementptr float, float* %276, i64 %3296
  store float %3299, float* %3300, align 4, !noalias !133
  %3301 = or i64 %3296, 1
  %3302 = getelementptr float, float* %265, i64 %3301
  %3303 = load float, float* %3302, align 4, !noalias !133
  %3304 = fmul float %3303, %3086
  %3305 = getelementptr float, float* %276, i64 %3301
  store float %3304, float* %3305, align 4, !noalias !133
  %3306 = add nuw nsw i64 %3079, 126
  %3307 = getelementptr float, float* %265, i64 %3306
  %3308 = getelementptr float, float* %276, i64 %3306
  %3309 = bitcast float* %3307 to <8 x float>*
  %3310 = load <8 x float>, <8 x float>* %3309, align 4, !noalias !133
  %3311 = fmul <8 x float> %3310, %3090
  %3312 = bitcast float* %3308 to <8 x float>*
  store <8 x float> %3311, <8 x float>* %3312, align 4, !noalias !133
  %3313 = add nuw nsw i64 %3079, 134
  %3314 = getelementptr float, float* %265, i64 %3313
  %3315 = getelementptr float, float* %276, i64 %3313
  %3316 = bitcast float* %3314 to <4 x float>*
  %3317 = load <4 x float>, <4 x float>* %3316, align 4, !noalias !133
  %3318 = fmul <4 x float> %3317, %3092
  %3319 = bitcast float* %3315 to <4 x float>*
  store <4 x float> %3318, <4 x float>* %3319, align 4, !noalias !133
  %3320 = add nuw nsw i64 %3079, 138
  %3321 = getelementptr float, float* %265, i64 %3320
  %3322 = load float, float* %3321, align 4, !noalias !133
  %3323 = fmul float %3322, %3086
  %3324 = getelementptr float, float* %276, i64 %3320
  store float %3323, float* %3324, align 4, !noalias !133
  %3325 = or i64 %3320, 1
  %3326 = getelementptr float, float* %265, i64 %3325
  %3327 = load float, float* %3326, align 4, !noalias !133
  %3328 = fmul float %3327, %3086
  %3329 = getelementptr float, float* %276, i64 %3325
  store float %3328, float* %3329, align 4, !noalias !133
  %3330 = add nuw nsw i64 %3079, 140
  %3331 = getelementptr float, float* %265, i64 %3330
  %3332 = getelementptr float, float* %276, i64 %3330
  %3333 = bitcast float* %3331 to <8 x float>*
  %3334 = load <8 x float>, <8 x float>* %3333, align 4, !noalias !133
  %3335 = fmul <8 x float> %3334, %3090
  %3336 = bitcast float* %3332 to <8 x float>*
  store <8 x float> %3335, <8 x float>* %3336, align 4, !noalias !133
  %3337 = add nuw nsw i64 %3079, 148
  %3338 = getelementptr float, float* %265, i64 %3337
  %3339 = getelementptr float, float* %276, i64 %3337
  %3340 = bitcast float* %3338 to <4 x float>*
  %3341 = load <4 x float>, <4 x float>* %3340, align 4, !noalias !133
  %3342 = fmul <4 x float> %3341, %3092
  %3343 = bitcast float* %3339 to <4 x float>*
  store <4 x float> %3342, <4 x float>* %3343, align 4, !noalias !133
  %3344 = add nuw nsw i64 %3079, 152
  %3345 = getelementptr float, float* %265, i64 %3344
  %3346 = load float, float* %3345, align 4, !noalias !133
  %3347 = fmul float %3346, %3086
  %3348 = getelementptr float, float* %276, i64 %3344
  store float %3347, float* %3348, align 4, !noalias !133
  %3349 = or i64 %3344, 1
  %3350 = getelementptr float, float* %265, i64 %3349
  %3351 = load float, float* %3350, align 4, !noalias !133
  %3352 = fmul float %3351, %3086
  %3353 = getelementptr float, float* %276, i64 %3349
  store float %3352, float* %3353, align 4, !noalias !133
  %3354 = add nuw nsw i64 %3079, 154
  %3355 = getelementptr float, float* %265, i64 %3354
  %3356 = getelementptr float, float* %276, i64 %3354
  %3357 = bitcast float* %3355 to <8 x float>*
  %3358 = load <8 x float>, <8 x float>* %3357, align 4, !noalias !133
  %3359 = fmul <8 x float> %3358, %3090
  %3360 = bitcast float* %3356 to <8 x float>*
  store <8 x float> %3359, <8 x float>* %3360, align 4, !noalias !133
  %3361 = add nuw nsw i64 %3079, 162
  %3362 = getelementptr float, float* %265, i64 %3361
  %3363 = getelementptr float, float* %276, i64 %3361
  %3364 = bitcast float* %3362 to <4 x float>*
  %3365 = load <4 x float>, <4 x float>* %3364, align 4, !noalias !133
  %3366 = fmul <4 x float> %3365, %3092
  %3367 = bitcast float* %3363 to <4 x float>*
  store <4 x float> %3366, <4 x float>* %3367, align 4, !noalias !133
  %3368 = add nuw nsw i64 %3079, 166
  %3369 = getelementptr float, float* %265, i64 %3368
  %3370 = load float, float* %3369, align 4, !noalias !133
  %3371 = fmul float %3370, %3086
  %3372 = getelementptr float, float* %276, i64 %3368
  store float %3371, float* %3372, align 4, !noalias !133
  %3373 = or i64 %3368, 1
  %3374 = getelementptr float, float* %265, i64 %3373
  %3375 = load float, float* %3374, align 4, !noalias !133
  %3376 = fmul float %3375, %3086
  %3377 = getelementptr float, float* %276, i64 %3373
  store float %3376, float* %3377, align 4, !noalias !133
  %3378 = add nuw nsw i64 %3079, 168
  %3379 = getelementptr float, float* %265, i64 %3378
  %3380 = getelementptr float, float* %276, i64 %3378
  %3381 = bitcast float* %3379 to <8 x float>*
  %3382 = load <8 x float>, <8 x float>* %3381, align 4, !noalias !133
  %3383 = fmul <8 x float> %3382, %3090
  %3384 = bitcast float* %3380 to <8 x float>*
  store <8 x float> %3383, <8 x float>* %3384, align 4, !noalias !133
  %3385 = add nuw nsw i64 %3079, 176
  %3386 = getelementptr float, float* %265, i64 %3385
  %3387 = getelementptr float, float* %276, i64 %3385
  %3388 = bitcast float* %3386 to <4 x float>*
  %3389 = load <4 x float>, <4 x float>* %3388, align 4, !noalias !133
  %3390 = fmul <4 x float> %3389, %3092
  %3391 = bitcast float* %3387 to <4 x float>*
  store <4 x float> %3390, <4 x float>* %3391, align 4, !noalias !133
  %3392 = add nuw nsw i64 %3079, 180
  %3393 = getelementptr float, float* %265, i64 %3392
  %3394 = load float, float* %3393, align 4, !noalias !133
  %3395 = fmul float %3394, %3086
  %3396 = getelementptr float, float* %276, i64 %3392
  store float %3395, float* %3396, align 4, !noalias !133
  %3397 = or i64 %3392, 1
  %3398 = getelementptr float, float* %265, i64 %3397
  %3399 = load float, float* %3398, align 4, !noalias !133
  %3400 = fmul float %3399, %3086
  %3401 = getelementptr float, float* %276, i64 %3397
  store float %3400, float* %3401, align 4, !noalias !133
  %3402 = add nuw nsw i64 %3079, 182
  %3403 = getelementptr float, float* %265, i64 %3402
  %3404 = getelementptr float, float* %276, i64 %3402
  %3405 = bitcast float* %3403 to <8 x float>*
  %3406 = load <8 x float>, <8 x float>* %3405, align 4, !noalias !133
  %3407 = fmul <8 x float> %3406, %3090
  %3408 = bitcast float* %3404 to <8 x float>*
  store <8 x float> %3407, <8 x float>* %3408, align 4, !noalias !133
  %3409 = add nuw nsw i64 %3079, 190
  %3410 = getelementptr float, float* %265, i64 %3409
  %3411 = getelementptr float, float* %276, i64 %3409
  %3412 = bitcast float* %3410 to <4 x float>*
  %3413 = load <4 x float>, <4 x float>* %3412, align 4, !noalias !133
  %3414 = fmul <4 x float> %3413, %3092
  %3415 = bitcast float* %3411 to <4 x float>*
  store <4 x float> %3414, <4 x float>* %3415, align 4, !noalias !133
  %3416 = add nuw nsw i64 %3079, 194
  %3417 = getelementptr float, float* %265, i64 %3416
  %3418 = load float, float* %3417, align 4, !noalias !133
  %3419 = fmul float %3418, %3086
  %3420 = getelementptr float, float* %276, i64 %3416
  store float %3419, float* %3420, align 4, !noalias !133
  %3421 = or i64 %3416, 1
  %3422 = getelementptr float, float* %265, i64 %3421
  %3423 = load float, float* %3422, align 4, !noalias !133
  %3424 = fmul float %3423, %3086
  %3425 = getelementptr float, float* %276, i64 %3421
  store float %3424, float* %3425, align 4, !noalias !133
  %3426 = add nuw nsw i64 %3078, 1
  %exitcond466.not.i = icmp eq i64 %3426, 672
  br i1 %exitcond466.not.i, label %exit452.i, label %cond453.preheader.i

exit452.i:                                        ; preds = %cond453.preheader.i
  %3427 = alloca [3 x i8*], align 8
  %3428 = alloca [3 x i64], align 16
  %3429 = alloca [8 x i64], align 8
  %3430 = alloca [3 x i8], align 1
  %.sub262.i = getelementptr inbounds [3 x i8], [3 x i8]* %3430, i64 0, i64 0
  %.sub261.i = getelementptr inbounds [8 x i64], [8 x i64]* %3429, i64 0, i64 0
  %.sub260.i = getelementptr inbounds [3 x i64], [3 x i64]* %3428, i64 0, i64 0
  %.sub259.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %3427, i64 0, i64 0
  store i8* %malloccall56.i, i8** %.sub259.i, align 8, !noalias !133
  store i8 6, i8* %.sub262.i, align 1, !noalias !133
  %3431 = bitcast [8 x i64]* %3429 to <4 x i64>*
  store <4 x i64> <i64 1, i64 112, i64 14, i64 14>, <4 x i64>* %3431, align 8, !noalias !133
  %3432 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3427, i64 0, i64 1
  store i8* %malloccall38.i, i8** %3432, align 8, !noalias !133
  %3433 = getelementptr inbounds [3 x i8], [3 x i8]* %3430, i64 0, i64 1
  store i8 6, i8* %3433, align 1, !noalias !133
  %3434 = bitcast [3 x i64]* %3428 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %3434, align 16, !noalias !133
  %3435 = getelementptr inbounds [8 x i64], [8 x i64]* %3429, i64 0, i64 4
  %3436 = bitcast i64* %3435 to <4 x i64>*
  store <4 x i64> <i64 1, i64 672, i64 14, i64 14>, <4 x i64>* %3436, align 8, !noalias !133
  %3437 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3427, i64 0, i64 2
  %3438 = bitcast i8** %3437 to float**
  store float* %158, float** %3438, align 8, !noalias !133
  %3439 = getelementptr inbounds [3 x i8], [3 x i8]* %3430, i64 0, i64 2
  store i8 6, i8* %3439, align 1, !noalias !133
  %3440 = getelementptr inbounds [3 x i64], [3 x i64]* %3428, i64 0, i64 2
  store i64 0, i64* %3440, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub259.i, i64* nonnull %.sub260.i, i64* nonnull %.sub261.i, i8* nonnull %.sub262.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond468.preheader.i

cond468.preheader.i:                              ; preds = %exit470.i, %exit452.i
  %3441 = phi i64 [ 0, %exit452.i ], [ %3532, %exit470.i ]
  %3442 = mul nuw nsw i64 %3441, 196
  %3443 = add nuw nsw i64 %3442, 8
  %3444 = add nuw nsw i64 %3442, 12
  br label %cond471.preheader.i

exit467.i:                                        ; preds = %exit470.i
  %3445 = alloca [3 x i8*], align 8
  %3446 = alloca [3 x i64], align 16
  %3447 = alloca [8 x i64], align 8
  %3448 = alloca [3 x i8], align 1
  %.sub267.i = getelementptr inbounds [3 x i8], [3 x i8]* %3448, i64 0, i64 0
  %.sub266.i = getelementptr inbounds [8 x i64], [8 x i64]* %3447, i64 0, i64 0
  %.sub265.i = getelementptr inbounds [3 x i64], [3 x i64]* %3446, i64 0, i64 0
  %.sub264.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %3445, i64 0, i64 0
  store i8* %malloccall8.i, i8** %.sub264.i, align 8, !noalias !133
  store i8 6, i8* %.sub267.i, align 1, !noalias !133
  %3449 = bitcast [8 x i64]* %3447 to <4 x i64>*
  store <4 x i64> <i64 1, i64 672, i64 14, i64 14>, <4 x i64>* %3449, align 8, !noalias !133
  %3450 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3445, i64 0, i64 1
  store i8* %malloccall9.i, i8** %3450, align 8, !noalias !133
  %3451 = getelementptr inbounds [3 x i8], [3 x i8]* %3448, i64 0, i64 1
  store i8 6, i8* %3451, align 1, !noalias !133
  %3452 = bitcast [3 x i64]* %3446 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %3452, align 16, !noalias !133
  %3453 = getelementptr inbounds [8 x i64], [8 x i64]* %3447, i64 0, i64 4
  %3454 = bitcast i64* %3453 to <4 x i64>*
  store <4 x i64> <i64 1, i64 112, i64 14, i64 14>, <4 x i64>* %3454, align 8, !noalias !133
  %3455 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3445, i64 0, i64 2
  %3456 = bitcast i8** %3455 to float**
  store float* %161, float** %3456, align 8, !noalias !133
  %3457 = getelementptr inbounds [3 x i8], [3 x i8]* %3448, i64 0, i64 2
  store i8 6, i8* %3457, align 1, !noalias !133
  %3458 = getelementptr inbounds [3 x i64], [3 x i64]* %3446, i64 0, i64 2
  store i64 0, i64* %3458, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub264.i, i64* nonnull %.sub265.i, i64* nonnull %.sub266.i, i8* nonnull %.sub267.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond483.preheader.i

cond471.preheader.i:                              ; preds = %cond471.preheader.i, %cond468.preheader.i
  %3459 = phi i64 [ 0, %cond468.preheader.i ], [ %3531, %cond471.preheader.i ]
  %3460 = mul nuw nsw i64 %3459, 14
  %3461 = add nuw nsw i64 %3460, %3442
  %3462 = getelementptr float, float* %289, i64 %3461
  %3463 = getelementptr float, float* %307, i64 %3461
  %3464 = getelementptr float, float* %251, i64 %3461
  %3465 = bitcast float* %3462 to <8 x float>*
  %3466 = load <8 x float>, <8 x float>* %3465, align 4, !noalias !133
  %3467 = bitcast float* %3463 to <8 x float>*
  %3468 = load <8 x float>, <8 x float>* %3467, align 4, !noalias !133
  %3469 = fadd <8 x float> %3466, %3468
  %3470 = bitcast float* %3464 to <8 x float>*
  store <8 x float> %3469, <8 x float>* %3470, align 4, !noalias !133
  %3471 = add nuw nsw i64 %3443, %3460
  %3472 = getelementptr float, float* %289, i64 %3471
  %3473 = getelementptr float, float* %307, i64 %3471
  %3474 = getelementptr float, float* %251, i64 %3471
  %3475 = bitcast float* %3472 to <4 x float>*
  %3476 = load <4 x float>, <4 x float>* %3475, align 4, !noalias !133
  %3477 = bitcast float* %3473 to <4 x float>*
  %3478 = load <4 x float>, <4 x float>* %3477, align 4, !noalias !133
  %3479 = fadd <4 x float> %3476, %3478
  %3480 = bitcast float* %3474 to <4 x float>*
  store <4 x float> %3479, <4 x float>* %3480, align 4, !noalias !133
  %3481 = add nuw nsw i64 %3444, %3460
  %3482 = getelementptr float, float* %289, i64 %3481
  %3483 = load float, float* %3482, align 4, !noalias !133
  %3484 = getelementptr float, float* %307, i64 %3481
  %3485 = load float, float* %3484, align 4, !noalias !133
  %3486 = fadd float %3483, %3485
  %3487 = getelementptr float, float* %251, i64 %3481
  store float %3486, float* %3487, align 4, !noalias !133
  %3488 = or i64 %3481, 1
  %3489 = getelementptr float, float* %289, i64 %3488
  %3490 = load float, float* %3489, align 4, !noalias !133
  %3491 = getelementptr float, float* %307, i64 %3488
  %3492 = load float, float* %3491, align 4, !noalias !133
  %3493 = fadd float %3490, %3492
  %3494 = getelementptr float, float* %251, i64 %3488
  store float %3493, float* %3494, align 4, !noalias !133
  %3495 = or i64 %3459, 1
  %3496 = mul nuw nsw i64 %3495, 14
  %3497 = add nuw nsw i64 %3496, %3442
  %3498 = getelementptr float, float* %289, i64 %3497
  %3499 = getelementptr float, float* %307, i64 %3497
  %3500 = getelementptr float, float* %251, i64 %3497
  %3501 = bitcast float* %3498 to <8 x float>*
  %3502 = load <8 x float>, <8 x float>* %3501, align 4, !noalias !133
  %3503 = bitcast float* %3499 to <8 x float>*
  %3504 = load <8 x float>, <8 x float>* %3503, align 4, !noalias !133
  %3505 = fadd <8 x float> %3502, %3504
  %3506 = bitcast float* %3500 to <8 x float>*
  store <8 x float> %3505, <8 x float>* %3506, align 4, !noalias !133
  %3507 = add nuw nsw i64 %3443, %3496
  %3508 = getelementptr float, float* %289, i64 %3507
  %3509 = getelementptr float, float* %307, i64 %3507
  %3510 = getelementptr float, float* %251, i64 %3507
  %3511 = bitcast float* %3508 to <4 x float>*
  %3512 = load <4 x float>, <4 x float>* %3511, align 4, !noalias !133
  %3513 = bitcast float* %3509 to <4 x float>*
  %3514 = load <4 x float>, <4 x float>* %3513, align 4, !noalias !133
  %3515 = fadd <4 x float> %3512, %3514
  %3516 = bitcast float* %3510 to <4 x float>*
  store <4 x float> %3515, <4 x float>* %3516, align 4, !noalias !133
  %3517 = add nuw nsw i64 %3444, %3496
  %3518 = getelementptr float, float* %289, i64 %3517
  %3519 = load float, float* %3518, align 4, !noalias !133
  %3520 = getelementptr float, float* %307, i64 %3517
  %3521 = load float, float* %3520, align 4, !noalias !133
  %3522 = fadd float %3519, %3521
  %3523 = getelementptr float, float* %251, i64 %3517
  store float %3522, float* %3523, align 4, !noalias !133
  %3524 = or i64 %3517, 1
  %3525 = getelementptr float, float* %289, i64 %3524
  %3526 = load float, float* %3525, align 4, !noalias !133
  %3527 = getelementptr float, float* %307, i64 %3524
  %3528 = load float, float* %3527, align 4, !noalias !133
  %3529 = fadd float %3526, %3528
  %3530 = getelementptr float, float* %251, i64 %3524
  store float %3529, float* %3530, align 4, !noalias !133
  %3531 = add nuw nsw i64 %3459, 2
  %exitcond461.not.1.i = icmp eq i64 %3531, 14
  br i1 %exitcond461.not.1.i, label %exit470.i, label %cond471.preheader.i

exit470.i:                                        ; preds = %cond471.preheader.i
  %3532 = add nuw nsw i64 %3441, 1
  %exitcond462.not.i = icmp eq i64 %3532, 112
  br i1 %exitcond462.not.i, label %exit467.i, label %cond468.preheader.i

cond483.preheader.i:                              ; preds = %exit485.i, %exit467.i
  %3533 = phi i64 [ 0, %exit467.i ], [ %3602, %exit485.i ]
  %3534 = mul nuw nsw i64 %3533, 196
  %3535 = add nuw nsw i64 %3534, 8
  %3536 = add nuw nsw i64 %3534, 12
  br label %cond486.preheader.i

exit482.i:                                        ; preds = %exit485.i
  %3537 = alloca [3 x i8*], align 8
  %3538 = alloca [3 x i64], align 16
  %3539 = alloca [8 x i64], align 8
  %3540 = alloca [3 x i8], align 1
  %.sub272.i = getelementptr inbounds [3 x i8], [3 x i8]* %3540, i64 0, i64 0
  %.sub271.i = getelementptr inbounds [8 x i64], [8 x i64]* %3539, i64 0, i64 0
  %.sub270.i = getelementptr inbounds [3 x i64], [3 x i64]* %3538, i64 0, i64 0
  %.sub269.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %3537, i64 0, i64 0
  store i8* %malloccall7.i, i8** %.sub269.i, align 8, !noalias !133
  store i8 6, i8* %.sub272.i, align 1, !noalias !133
  %3541 = bitcast [8 x i64]* %3539 to <4 x i64>*
  store <4 x i64> <i64 1, i64 672, i64 7, i64 7>, <4 x i64>* %3541, align 8, !noalias !133
  %3542 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3537, i64 0, i64 1
  store i8* %malloccall61.i, i8** %3542, align 8, !noalias !133
  %3543 = getelementptr inbounds [3 x i8], [3 x i8]* %3540, i64 0, i64 1
  store i8 6, i8* %3543, align 1, !noalias !133
  %3544 = bitcast [3 x i64]* %3538 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %3544, align 16, !noalias !133
  %3545 = getelementptr inbounds [8 x i64], [8 x i64]* %3539, i64 0, i64 4
  %3546 = bitcast i64* %3545 to <4 x i64>*
  store <4 x i64> <i64 1, i64 672, i64 14, i64 14>, <4 x i64>* %3546, align 8, !noalias !133
  %3547 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3537, i64 0, i64 2
  %3548 = bitcast i8** %3547 to float**
  store float* %164, float** %3548, align 8, !noalias !133
  %3549 = getelementptr inbounds [3 x i8], [3 x i8]* %3540, i64 0, i64 2
  store i8 6, i8* %3549, align 1, !noalias !133
  %3550 = getelementptr inbounds [3 x i64], [3 x i64]* %3538, i64 0, i64 2
  store i64 0, i64* %3550, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub269.i, i64* nonnull %.sub270.i, i64* nonnull %.sub271.i, i8* nonnull %.sub272.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond498.preheader.i

cond486.preheader.i:                              ; preds = %cond486.preheader.i, %cond483.preheader.i
  %3551 = phi i64 [ 0, %cond483.preheader.i ], [ %3601, %cond486.preheader.i ]
  %3552 = mul nuw nsw i64 %3551, 14
  %3553 = add nuw nsw i64 %3552, %3534
  %3554 = getelementptr float, float* %250, i64 %3553
  %3555 = getelementptr float, float* %293, i64 %3553
  %3556 = bitcast float* %3554 to <8 x float>*
  %3557 = load <8 x float>, <8 x float>* %3556, align 4, !noalias !133
  %3558 = fadd <8 x float> %3557, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %3559 = fcmp olt <8 x float> %3558, zeroinitializer
  %3560 = select <8 x i1> %3559, <8 x float> zeroinitializer, <8 x float> %3558
  %3561 = fcmp ogt <8 x float> %3560, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %3562 = select <8 x i1> %3561, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %3560
  %3563 = fmul <8 x float> %3557, %3562
  %3564 = fdiv <8 x float> %3563, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %3565 = bitcast float* %3555 to <8 x float>*
  store <8 x float> %3564, <8 x float>* %3565, align 4, !noalias !133
  %3566 = add nuw nsw i64 %3535, %3552
  %3567 = getelementptr float, float* %250, i64 %3566
  %3568 = getelementptr float, float* %293, i64 %3566
  %3569 = bitcast float* %3567 to <4 x float>*
  %3570 = load <4 x float>, <4 x float>* %3569, align 4, !noalias !133
  %3571 = fadd <4 x float> %3570, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %3572 = fcmp olt <4 x float> %3571, zeroinitializer
  %3573 = select <4 x i1> %3572, <4 x float> zeroinitializer, <4 x float> %3571
  %3574 = fcmp ogt <4 x float> %3573, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %3575 = select <4 x i1> %3574, <4 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <4 x float> %3573
  %3576 = fmul <4 x float> %3570, %3575
  %3577 = fdiv <4 x float> %3576, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %3578 = bitcast float* %3568 to <4 x float>*
  store <4 x float> %3577, <4 x float>* %3578, align 4, !noalias !133
  %3579 = add nuw nsw i64 %3536, %3552
  %3580 = getelementptr float, float* %250, i64 %3579
  %3581 = load float, float* %3580, align 4, !noalias !133
  %3582 = fadd float %3581, 3.000000e+00
  %3583 = fcmp olt float %3582, 0.000000e+00
  %3584 = select i1 %3583, float 0.000000e+00, float %3582
  %3585 = fcmp ogt float %3584, 6.000000e+00
  %3586 = select i1 %3585, float 6.000000e+00, float %3584
  %3587 = fmul float %3581, %3586
  %3588 = fdiv float %3587, 6.000000e+00
  %3589 = getelementptr float, float* %293, i64 %3579
  store float %3588, float* %3589, align 4, !noalias !133
  %3590 = or i64 %3579, 1
  %3591 = getelementptr float, float* %250, i64 %3590
  %3592 = load float, float* %3591, align 4, !noalias !133
  %3593 = fadd float %3592, 3.000000e+00
  %3594 = fcmp olt float %3593, 0.000000e+00
  %3595 = select i1 %3594, float 0.000000e+00, float %3593
  %3596 = fcmp ogt float %3595, 6.000000e+00
  %3597 = select i1 %3596, float 6.000000e+00, float %3595
  %3598 = fmul float %3592, %3597
  %3599 = fdiv float %3598, 6.000000e+00
  %3600 = getelementptr float, float* %293, i64 %3590
  store float %3599, float* %3600, align 4, !noalias !133
  %3601 = add nuw nsw i64 %3551, 1
  %exitcond457.not.i = icmp eq i64 %3601, 14
  br i1 %exitcond457.not.i, label %exit485.i, label %cond486.preheader.i

exit485.i:                                        ; preds = %cond486.preheader.i
  %3602 = add nuw nsw i64 %3533, 1
  %exitcond458.not.i = icmp eq i64 %3602, 672
  br i1 %exitcond458.not.i, label %exit482.i, label %cond483.preheader.i

cond498.preheader.i:                              ; preds = %exit500.i, %exit482.i
  %3603 = phi i64 [ 0, %exit482.i ], [ %3693, %exit500.i ]
  %3604 = mul nuw nsw i64 %3603, 49
  %3605 = add nuw nsw i64 %3604, 4
  br label %cond501.preheader.i

exit497.i:                                        ; preds = %exit500.i
  %3606 = alloca [2 x i8*], align 8
  %3607 = alloca <2 x i64>, align 16
  %3608 = alloca [8 x i64], align 8
  %3609 = alloca [2 x i8], align 1
  %3610 = alloca <2 x i64>, align 16
  %.sub278.i = getelementptr inbounds <2 x i64>, <2 x i64>* %3610, i64 0, i64 0
  %.sub277.i = getelementptr inbounds [2 x i8], [2 x i8]* %3609, i64 0, i64 0
  %.sub276.i = getelementptr inbounds [8 x i64], [8 x i64]* %3608, i64 0, i64 0
  %.sub275.i = getelementptr inbounds <2 x i64>, <2 x i64>* %3607, i64 0, i64 0
  %.sub274.i = getelementptr inbounds [2 x i8*], [2 x i8*]* %3606, i64 0, i64 0
  store i8* %malloccall34.i, i8** %.sub274.i, align 8, !noalias !133
  store i8 6, i8* %.sub277.i, align 1, !noalias !133
  %3611 = bitcast [8 x i64]* %3608 to <4 x i64>*
  store <4 x i64> <i64 1, i64 672, i64 1, i64 1>, <4 x i64>* %3611, align 8, !noalias !133
  %3612 = getelementptr inbounds [2 x i8*], [2 x i8*]* %3606, i64 0, i64 1
  store i8* %malloccall5.i, i8** %3612, align 8, !noalias !133
  %3613 = getelementptr inbounds [2 x i8], [2 x i8]* %3609, i64 0, i64 1
  store i8 6, i8* %3613, align 1, !noalias !133
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %3607, align 16, !noalias !133
  %3614 = getelementptr inbounds [8 x i64], [8 x i64]* %3608, i64 0, i64 4
  %3615 = bitcast i64* %3614 to <4 x i64>*
  store <4 x i64> <i64 1, i64 672, i64 7, i64 7>, <4 x i64>* %3615, align 8, !noalias !133
  store <2 x i64> <i64 1, i64 1>, <2 x i64>* %3610, align 16, !noalias !133
  call void @nnc_aten_adaptive_avg_pool2d(i64 2, i8** nonnull %.sub274.i, i64* nonnull %.sub275.i, i64* nonnull %.sub276.i, i8* nonnull %.sub277.i, i64 2, i64* nonnull %.sub278.i) #0, !noalias !135
  %3616 = alloca [3 x i8*], align 8
  %3617 = alloca [3 x i64], align 16
  %3618 = alloca [8 x i64], align 8
  %3619 = alloca [3 x i8], align 1
  %.sub282.i = getelementptr inbounds [3 x i8], [3 x i8]* %3619, i64 0, i64 0
  %.sub281.i = getelementptr inbounds [8 x i64], [8 x i64]* %3618, i64 0, i64 0
  %.sub280.i = getelementptr inbounds [3 x i64], [3 x i64]* %3617, i64 0, i64 0
  %.sub279.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %3616, i64 0, i64 0
  store i8* %malloccall67.i, i8** %.sub279.i, align 8, !noalias !133
  store i8 6, i8* %.sub282.i, align 1, !noalias !133
  %3620 = bitcast [8 x i64]* %3618 to <4 x i64>*
  store <4 x i64> <i64 1, i64 168, i64 1, i64 1>, <4 x i64>* %3620, align 8, !noalias !133
  %3621 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3616, i64 0, i64 1
  store i8* %malloccall34.i, i8** %3621, align 8, !noalias !133
  %3622 = getelementptr inbounds [3 x i8], [3 x i8]* %3619, i64 0, i64 1
  store i8 6, i8* %3622, align 1, !noalias !133
  %3623 = bitcast [3 x i64]* %3617 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %3623, align 16, !noalias !133
  %3624 = getelementptr inbounds [8 x i64], [8 x i64]* %3618, i64 0, i64 4
  %3625 = bitcast i64* %3624 to <4 x i64>*
  store <4 x i64> <i64 1, i64 672, i64 1, i64 1>, <4 x i64>* %3625, align 8, !noalias !133
  %3626 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3616, i64 0, i64 2
  %3627 = bitcast i8** %3626 to float**
  store float* %167, float** %3627, align 8, !noalias !133
  %3628 = getelementptr inbounds [3 x i8], [3 x i8]* %3619, i64 0, i64 2
  store i8 6, i8* %3628, align 1, !noalias !133
  %3629 = getelementptr inbounds [3 x i64], [3 x i64]* %3617, i64 0, i64 2
  store i64 0, i64* %3629, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub279.i, i64* nonnull %.sub280.i, i64* nonnull %.sub281.i, i8* nonnull %.sub282.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %3630 = alloca [3 x i8*], align 8
  %3631 = alloca [3 x i64], align 16
  %3632 = alloca [8 x i64], align 8
  %3633 = alloca [3 x i8], align 1
  %.sub287.i = getelementptr inbounds [3 x i8], [3 x i8]* %3633, i64 0, i64 0
  %.sub286.i = getelementptr inbounds [8 x i64], [8 x i64]* %3632, i64 0, i64 0
  %.sub285.i = getelementptr inbounds [3 x i64], [3 x i64]* %3631, i64 0, i64 0
  %.sub284.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %3630, i64 0, i64 0
  store i8* %malloccall88.i, i8** %.sub284.i, align 8, !noalias !133
  store i8 6, i8* %.sub287.i, align 1, !noalias !133
  %3634 = bitcast [8 x i64]* %3632 to <4 x i64>*
  store <4 x i64> <i64 1, i64 672, i64 1, i64 1>, <4 x i64>* %3634, align 8, !noalias !133
  %3635 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3630, i64 0, i64 1
  store i8* %malloccall67.i, i8** %3635, align 8, !noalias !133
  %3636 = getelementptr inbounds [3 x i8], [3 x i8]* %3633, i64 0, i64 1
  store i8 6, i8* %3636, align 1, !noalias !133
  %3637 = bitcast [3 x i64]* %3631 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %3637, align 16, !noalias !133
  %3638 = getelementptr inbounds [8 x i64], [8 x i64]* %3632, i64 0, i64 4
  %3639 = bitcast i64* %3638 to <4 x i64>*
  store <4 x i64> <i64 1, i64 168, i64 1, i64 1>, <4 x i64>* %3639, align 8, !noalias !133
  %3640 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3630, i64 0, i64 2
  %3641 = bitcast i8** %3640 to float**
  store float* %170, float** %3641, align 8, !noalias !133
  %3642 = getelementptr inbounds [3 x i8], [3 x i8]* %3633, i64 0, i64 2
  store i8 6, i8* %3642, align 1, !noalias !133
  %3643 = getelementptr inbounds [3 x i64], [3 x i64]* %3631, i64 0, i64 2
  store i64 0, i64* %3643, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub284.i, i64* nonnull %.sub285.i, i64* nonnull %.sub286.i, i8* nonnull %.sub287.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond510.preheader.i

cond501.preheader.i:                              ; preds = %cond501.preheader.i, %cond498.preheader.i
  %3644 = phi i64 [ 0, %cond498.preheader.i ], [ %3692, %cond501.preheader.i ]
  %3645 = mul nuw nsw i64 %3644, 7
  %3646 = add nuw nsw i64 %3645, %3604
  %3647 = getelementptr float, float* %249, i64 %3646
  %3648 = getelementptr float, float* %247, i64 %3646
  %3649 = bitcast float* %3647 to <4 x float>*
  %3650 = load <4 x float>, <4 x float>* %3649, align 4, !noalias !133
  %3651 = fadd <4 x float> %3650, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %3652 = fcmp olt <4 x float> %3651, zeroinitializer
  %3653 = select <4 x i1> %3652, <4 x float> zeroinitializer, <4 x float> %3651
  %3654 = fcmp ogt <4 x float> %3653, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %3655 = select <4 x i1> %3654, <4 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <4 x float> %3653
  %3656 = fmul <4 x float> %3650, %3655
  %3657 = fdiv <4 x float> %3656, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %3658 = bitcast float* %3648 to <4 x float>*
  store <4 x float> %3657, <4 x float>* %3658, align 4, !noalias !133
  %3659 = add nuw nsw i64 %3605, %3645
  %3660 = getelementptr float, float* %249, i64 %3659
  %3661 = load float, float* %3660, align 4, !noalias !133
  %3662 = fadd float %3661, 3.000000e+00
  %3663 = fcmp olt float %3662, 0.000000e+00
  %3664 = select i1 %3663, float 0.000000e+00, float %3662
  %3665 = fcmp ogt float %3664, 6.000000e+00
  %3666 = select i1 %3665, float 6.000000e+00, float %3664
  %3667 = fmul float %3661, %3666
  %3668 = fdiv float %3667, 6.000000e+00
  %3669 = getelementptr float, float* %247, i64 %3659
  store float %3668, float* %3669, align 4, !noalias !133
  %3670 = add nuw nsw i64 %3659, 1
  %3671 = getelementptr float, float* %249, i64 %3670
  %3672 = load float, float* %3671, align 4, !noalias !133
  %3673 = fadd float %3672, 3.000000e+00
  %3674 = fcmp olt float %3673, 0.000000e+00
  %3675 = select i1 %3674, float 0.000000e+00, float %3673
  %3676 = fcmp ogt float %3675, 6.000000e+00
  %3677 = select i1 %3676, float 6.000000e+00, float %3675
  %3678 = fmul float %3672, %3677
  %3679 = fdiv float %3678, 6.000000e+00
  %3680 = getelementptr float, float* %247, i64 %3670
  store float %3679, float* %3680, align 4, !noalias !133
  %3681 = add nuw nsw i64 %3659, 2
  %3682 = getelementptr float, float* %249, i64 %3681
  %3683 = load float, float* %3682, align 4, !noalias !133
  %3684 = fadd float %3683, 3.000000e+00
  %3685 = fcmp olt float %3684, 0.000000e+00
  %3686 = select i1 %3685, float 0.000000e+00, float %3684
  %3687 = fcmp ogt float %3686, 6.000000e+00
  %3688 = select i1 %3687, float 6.000000e+00, float %3686
  %3689 = fmul float %3683, %3688
  %3690 = fdiv float %3689, 6.000000e+00
  %3691 = getelementptr float, float* %247, i64 %3681
  store float %3690, float* %3691, align 4, !noalias !133
  %3692 = add nuw nsw i64 %3644, 1
  %exitcond453.not.i = icmp eq i64 %3692, 7
  br i1 %exitcond453.not.i, label %exit500.i, label %cond501.preheader.i

exit500.i:                                        ; preds = %cond501.preheader.i
  %3693 = add nuw nsw i64 %3603, 1
  %exitcond454.not.i = icmp eq i64 %3693, 672
  br i1 %exitcond454.not.i, label %exit497.i, label %cond498.preheader.i

cond510.preheader.i:                              ; preds = %cond510.preheader.i, %exit497.i
  %3694 = phi i64 [ 0, %exit497.i ], [ %3751, %cond510.preheader.i ]
  %3695 = mul nuw nsw i64 %3694, 49
  %3696 = getelementptr float, float* %313, i64 %3694
  %3697 = load float, float* %3696, align 4, !noalias !133
  %3698 = fadd float %3697, 3.000000e+00
  %3699 = fcmp olt float %3698, 0.000000e+00
  %3700 = select i1 %3699, float 0.000000e+00, float %3698
  %3701 = fcmp ogt float %3700, 6.000000e+00
  %.op384.i = fdiv float %3698, 6.000000e+00
  %.op383.i = select i1 %3699, float 0.000000e+00, float %.op384.i
  %3702 = select i1 %3701, float 1.000000e+00, float %.op383.i
  %3703 = getelementptr float, float* %247, i64 %3695
  %3704 = getelementptr float, float* %287, i64 %3695
  %3705 = bitcast float* %3703 to <8 x float>*
  %3706 = load <8 x float>, <8 x float>* %3705, align 4, !noalias !133
  %3707 = insertelement <8 x float> poison, float %3702, i32 0
  %3708 = shufflevector <8 x float> %3707, <8 x float> undef, <8 x i32> zeroinitializer
  %3709 = fmul <8 x float> %3706, %3708
  %3710 = bitcast float* %3704 to <8 x float>*
  store <8 x float> %3709, <8 x float>* %3710, align 4, !noalias !133
  %3711 = add nuw nsw i64 %3695, 8
  %3712 = getelementptr float, float* %247, i64 %3711
  %3713 = getelementptr float, float* %287, i64 %3711
  %3714 = bitcast float* %3712 to <8 x float>*
  %3715 = load <8 x float>, <8 x float>* %3714, align 4, !noalias !133
  %3716 = fmul <8 x float> %3715, %3708
  %3717 = bitcast float* %3713 to <8 x float>*
  store <8 x float> %3716, <8 x float>* %3717, align 4, !noalias !133
  %3718 = add nuw nsw i64 %3695, 16
  %3719 = getelementptr float, float* %247, i64 %3718
  %3720 = getelementptr float, float* %287, i64 %3718
  %3721 = bitcast float* %3719 to <8 x float>*
  %3722 = load <8 x float>, <8 x float>* %3721, align 4, !noalias !133
  %3723 = fmul <8 x float> %3722, %3708
  %3724 = bitcast float* %3720 to <8 x float>*
  store <8 x float> %3723, <8 x float>* %3724, align 4, !noalias !133
  %3725 = add nuw nsw i64 %3695, 24
  %3726 = getelementptr float, float* %247, i64 %3725
  %3727 = getelementptr float, float* %287, i64 %3725
  %3728 = bitcast float* %3726 to <8 x float>*
  %3729 = load <8 x float>, <8 x float>* %3728, align 4, !noalias !133
  %3730 = fmul <8 x float> %3729, %3708
  %3731 = bitcast float* %3727 to <8 x float>*
  store <8 x float> %3730, <8 x float>* %3731, align 4, !noalias !133
  %3732 = add nuw nsw i64 %3695, 32
  %3733 = getelementptr float, float* %247, i64 %3732
  %3734 = getelementptr float, float* %287, i64 %3732
  %3735 = bitcast float* %3733 to <8 x float>*
  %3736 = load <8 x float>, <8 x float>* %3735, align 4, !noalias !133
  %3737 = fmul <8 x float> %3736, %3708
  %3738 = bitcast float* %3734 to <8 x float>*
  store <8 x float> %3737, <8 x float>* %3738, align 4, !noalias !133
  %3739 = add nuw nsw i64 %3695, 40
  %3740 = getelementptr float, float* %247, i64 %3739
  %3741 = getelementptr float, float* %287, i64 %3739
  %3742 = bitcast float* %3740 to <8 x float>*
  %3743 = load <8 x float>, <8 x float>* %3742, align 4, !noalias !133
  %3744 = fmul <8 x float> %3743, %3708
  %3745 = bitcast float* %3741 to <8 x float>*
  store <8 x float> %3744, <8 x float>* %3745, align 4, !noalias !133
  %3746 = add nuw nsw i64 %3695, 48
  %3747 = getelementptr float, float* %247, i64 %3746
  %3748 = load float, float* %3747, align 4, !noalias !133
  %3749 = fmul float %3748, %3702
  %3750 = getelementptr float, float* %287, i64 %3746
  store float %3749, float* %3750, align 4, !noalias !133
  %3751 = add nuw nsw i64 %3694, 1
  %exitcond450.not.i = icmp eq i64 %3751, 672
  br i1 %exitcond450.not.i, label %exit509.i, label %cond510.preheader.i

exit509.i:                                        ; preds = %cond510.preheader.i
  %3752 = alloca [3 x i8*], align 8
  %3753 = alloca [3 x i64], align 16
  %3754 = alloca [8 x i64], align 8
  %3755 = alloca [3 x i8], align 1
  %.sub292.i = getelementptr inbounds [3 x i8], [3 x i8]* %3755, i64 0, i64 0
  %.sub291.i = getelementptr inbounds [8 x i64], [8 x i64]* %3754, i64 0, i64 0
  %.sub290.i = getelementptr inbounds [3 x i64], [3 x i64]* %3753, i64 0, i64 0
  %.sub289.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %3752, i64 0, i64 0
  store i8* %malloccall83.i, i8** %.sub289.i, align 8, !noalias !133
  store i8 6, i8* %.sub292.i, align 1, !noalias !133
  %3756 = bitcast [8 x i64]* %3754 to <4 x i64>*
  store <4 x i64> <i64 1, i64 160, i64 7, i64 7>, <4 x i64>* %3756, align 8, !noalias !133
  %3757 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3752, i64 0, i64 1
  store i8* %malloccall53.i, i8** %3757, align 8, !noalias !133
  %3758 = getelementptr inbounds [3 x i8], [3 x i8]* %3755, i64 0, i64 1
  store i8 6, i8* %3758, align 1, !noalias !133
  %3759 = bitcast [3 x i64]* %3753 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %3759, align 16, !noalias !133
  %3760 = getelementptr inbounds [8 x i64], [8 x i64]* %3754, i64 0, i64 4
  %3761 = bitcast i64* %3760 to <4 x i64>*
  store <4 x i64> <i64 1, i64 672, i64 7, i64 7>, <4 x i64>* %3761, align 8, !noalias !133
  %3762 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3752, i64 0, i64 2
  %3763 = bitcast i8** %3762 to float**
  store float* %173, float** %3763, align 8, !noalias !133
  %3764 = getelementptr inbounds [3 x i8], [3 x i8]* %3755, i64 0, i64 2
  store i8 6, i8* %3764, align 1, !noalias !133
  %3765 = getelementptr inbounds [3 x i64], [3 x i64]* %3753, i64 0, i64 2
  store i64 0, i64* %3765, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub289.i, i64* nonnull %.sub290.i, i64* nonnull %.sub291.i, i8* nonnull %.sub292.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  %3766 = alloca [3 x i8*], align 8
  %3767 = alloca [3 x i64], align 16
  %3768 = alloca [8 x i64], align 8
  %3769 = alloca [3 x i8], align 1
  %.sub297.i = getelementptr inbounds [3 x i8], [3 x i8]* %3769, i64 0, i64 0
  %.sub296.i = getelementptr inbounds [8 x i64], [8 x i64]* %3768, i64 0, i64 0
  %.sub295.i = getelementptr inbounds [3 x i64], [3 x i64]* %3767, i64 0, i64 0
  %.sub294.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %3766, i64 0, i64 0
  store i8* %malloccall6.i, i8** %.sub294.i, align 8, !noalias !133
  store i8 6, i8* %.sub297.i, align 1, !noalias !133
  %3770 = bitcast [8 x i64]* %3768 to <4 x i64>*
  store <4 x i64> <i64 1, i64 960, i64 7, i64 7>, <4 x i64>* %3770, align 8, !noalias !133
  %3771 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3766, i64 0, i64 1
  store i8* %malloccall83.i, i8** %3771, align 8, !noalias !133
  %3772 = getelementptr inbounds [3 x i8], [3 x i8]* %3769, i64 0, i64 1
  store i8 6, i8* %3772, align 1, !noalias !133
  %3773 = bitcast [3 x i64]* %3767 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %3773, align 16, !noalias !133
  %3774 = getelementptr inbounds [8 x i64], [8 x i64]* %3768, i64 0, i64 4
  %3775 = bitcast i64* %3774 to <4 x i64>*
  store <4 x i64> <i64 1, i64 160, i64 7, i64 7>, <4 x i64>* %3775, align 8, !noalias !133
  %3776 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3766, i64 0, i64 2
  %3777 = bitcast i8** %3776 to float**
  store float* %176, float** %3777, align 8, !noalias !133
  %3778 = getelementptr inbounds [3 x i8], [3 x i8]* %3769, i64 0, i64 2
  store i8 6, i8* %3778, align 1, !noalias !133
  %3779 = getelementptr inbounds [3 x i64], [3 x i64]* %3767, i64 0, i64 2
  store i64 0, i64* %3779, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub294.i, i64* nonnull %.sub295.i, i64* nonnull %.sub296.i, i8* nonnull %.sub297.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !135
  br label %cond522.preheader.i

cond522.preheader.i:                              ; preds = %exit524.i, %exit509.i
  %3780 = phi i64 [ 0, %exit509.i ], [ %3846, %exit524.i ]
  %3781 = mul nuw nsw i64 %3780, 49
  %3782 = add nuw nsw i64 %3781, 4
  br label %cond525.preheader.i

exit521.i:                                        ; preds = %exit524.i
  %3783 = alloca [3 x i8*], align 8
  %3784 = alloca [3 x i64], align 16
  %3785 = alloca [8 x i64], align 8
  %3786 = alloca [3 x i8], align 1
  %.sub302.i = getelementptr inbounds [3 x i8], [3 x i8]* %3786, i64 0, i64 0
  %.sub301.i = getelementptr inbounds [8 x i64], [8 x i64]* %3785, i64 0, i64 0
  %.sub300.i = getelementptr inbounds [3 x i64], [3 x i64]* %3784, i64 0, i64 0
  %.sub299.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %3783, i64 0, i64 0
  store i8* %malloccall43.i, i8** %.sub299.i, align 8, !noalias !133
  store i8 6, i8* %.sub302.i, align 1, !noalias !133
  %3787 = bitcast [8 x i64]* %3785 to <4 x i64>*
  store <4 x i64> <i64 1, i64 960, i64 7, i64 7>, <4 x i64>* %3787, align 8, !noalias !133
  %3788 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3783, i64 0, i64 1
  store i8* %malloccall2.i, i8** %3788, align 8, !noalias !133
  %3789 = getelementptr inbounds [3 x i8], [3 x i8]* %3786, i64 0, i64 1
  store i8 6, i8* %3789, align 1, !noalias !133
  %3790 = bitcast [3 x i64]* %3784 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %3790, align 16, !noalias !133
  %3791 = getelementptr inbounds [8 x i64], [8 x i64]* %3785, i64 0, i64 4
  %3792 = bitcast i64* %3791 to <4 x i64>*
  store <4 x i64> <i64 1, i64 960, i64 7, i64 7>, <4 x i64>* %3792, align 8, !noalias !133
  %3793 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3783, i64 0, i64 2
  %3794 = bitcast i8** %3793 to float**
  store float* %179, float** %3794, align 8, !noalias !133
  %3795 = getelementptr inbounds [3 x i8], [3 x i8]* %3786, i64 0, i64 2
  store i8 6, i8* %3795, align 1, !noalias !133
  %3796 = getelementptr inbounds [3 x i64], [3 x i64]* %3784, i64 0, i64 2
  store i64 0, i64* %3796, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub299.i, i64* nonnull %.sub300.i, i64* nonnull %.sub301.i, i8* nonnull %.sub302.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !136
  br label %cond534.preheader.i

cond525.preheader.i:                              ; preds = %cond525.preheader.i, %cond522.preheader.i
  %3797 = phi i64 [ 0, %cond522.preheader.i ], [ %3845, %cond525.preheader.i ]
  %3798 = mul nuw nsw i64 %3797, 7
  %3799 = add nuw nsw i64 %3798, %3781
  %3800 = getelementptr float, float* %248, i64 %3799
  %3801 = getelementptr float, float* %245, i64 %3799
  %3802 = bitcast float* %3800 to <4 x float>*
  %3803 = load <4 x float>, <4 x float>* %3802, align 4, !noalias !133
  %3804 = fadd <4 x float> %3803, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %3805 = fcmp olt <4 x float> %3804, zeroinitializer
  %3806 = select <4 x i1> %3805, <4 x float> zeroinitializer, <4 x float> %3804
  %3807 = fcmp ogt <4 x float> %3806, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %3808 = select <4 x i1> %3807, <4 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <4 x float> %3806
  %3809 = fmul <4 x float> %3803, %3808
  %3810 = fdiv <4 x float> %3809, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %3811 = bitcast float* %3801 to <4 x float>*
  store <4 x float> %3810, <4 x float>* %3811, align 4, !noalias !133
  %3812 = add nuw nsw i64 %3782, %3798
  %3813 = getelementptr float, float* %248, i64 %3812
  %3814 = load float, float* %3813, align 4, !noalias !133
  %3815 = fadd float %3814, 3.000000e+00
  %3816 = fcmp olt float %3815, 0.000000e+00
  %3817 = select i1 %3816, float 0.000000e+00, float %3815
  %3818 = fcmp ogt float %3817, 6.000000e+00
  %3819 = select i1 %3818, float 6.000000e+00, float %3817
  %3820 = fmul float %3814, %3819
  %3821 = fdiv float %3820, 6.000000e+00
  %3822 = getelementptr float, float* %245, i64 %3812
  store float %3821, float* %3822, align 4, !noalias !133
  %3823 = add nuw nsw i64 %3812, 1
  %3824 = getelementptr float, float* %248, i64 %3823
  %3825 = load float, float* %3824, align 4, !noalias !133
  %3826 = fadd float %3825, 3.000000e+00
  %3827 = fcmp olt float %3826, 0.000000e+00
  %3828 = select i1 %3827, float 0.000000e+00, float %3826
  %3829 = fcmp ogt float %3828, 6.000000e+00
  %3830 = select i1 %3829, float 6.000000e+00, float %3828
  %3831 = fmul float %3825, %3830
  %3832 = fdiv float %3831, 6.000000e+00
  %3833 = getelementptr float, float* %245, i64 %3823
  store float %3832, float* %3833, align 4, !noalias !133
  %3834 = add nuw nsw i64 %3812, 2
  %3835 = getelementptr float, float* %248, i64 %3834
  %3836 = load float, float* %3835, align 4, !noalias !133
  %3837 = fadd float %3836, 3.000000e+00
  %3838 = fcmp olt float %3837, 0.000000e+00
  %3839 = select i1 %3838, float 0.000000e+00, float %3837
  %3840 = fcmp ogt float %3839, 6.000000e+00
  %3841 = select i1 %3840, float 6.000000e+00, float %3839
  %3842 = fmul float %3836, %3841
  %3843 = fdiv float %3842, 6.000000e+00
  %3844 = getelementptr float, float* %245, i64 %3834
  store float %3843, float* %3844, align 4, !noalias !133
  %3845 = add nuw nsw i64 %3797, 1
  %exitcond445.not.i = icmp eq i64 %3845, 7
  br i1 %exitcond445.not.i, label %exit524.i, label %cond525.preheader.i

exit524.i:                                        ; preds = %cond525.preheader.i
  %3846 = add nuw nsw i64 %3780, 1
  %exitcond446.not.i = icmp eq i64 %3846, 960
  br i1 %exitcond446.not.i, label %exit521.i, label %cond522.preheader.i

cond534.preheader.i:                              ; preds = %exit536.i, %exit521.i
  %3847 = phi i64 [ 0, %exit521.i ], [ %3937, %exit536.i ]
  %3848 = mul nuw nsw i64 %3847, 49
  %3849 = add nuw nsw i64 %3848, 4
  br label %cond537.preheader.i

exit533.i:                                        ; preds = %exit536.i
  %3850 = alloca [2 x i8*], align 8
  %3851 = alloca <2 x i64>, align 16
  %3852 = alloca [8 x i64], align 8
  %3853 = alloca [2 x i8], align 1
  %3854 = alloca <2 x i64>, align 16
  %.sub308.i = getelementptr inbounds <2 x i64>, <2 x i64>* %3854, i64 0, i64 0
  %.sub307.i = getelementptr inbounds [2 x i8], [2 x i8]* %3853, i64 0, i64 0
  %.sub306.i = getelementptr inbounds [8 x i64], [8 x i64]* %3852, i64 0, i64 0
  %.sub305.i = getelementptr inbounds <2 x i64>, <2 x i64>* %3851, i64 0, i64 0
  %.sub304.i = getelementptr inbounds [2 x i8*], [2 x i8*]* %3850, i64 0, i64 0
  store i8* %malloccall51.i, i8** %.sub304.i, align 8, !noalias !133
  store i8 6, i8* %.sub307.i, align 1, !noalias !133
  %3855 = bitcast [8 x i64]* %3852 to <4 x i64>*
  store <4 x i64> <i64 1, i64 960, i64 1, i64 1>, <4 x i64>* %3855, align 8, !noalias !133
  %3856 = getelementptr inbounds [2 x i8*], [2 x i8*]* %3850, i64 0, i64 1
  store i8* %malloccall89.i, i8** %3856, align 8, !noalias !133
  %3857 = getelementptr inbounds [2 x i8], [2 x i8]* %3853, i64 0, i64 1
  store i8 6, i8* %3857, align 1, !noalias !133
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %3851, align 16, !noalias !133
  %3858 = getelementptr inbounds [8 x i64], [8 x i64]* %3852, i64 0, i64 4
  %3859 = bitcast i64* %3858 to <4 x i64>*
  store <4 x i64> <i64 1, i64 960, i64 7, i64 7>, <4 x i64>* %3859, align 8, !noalias !133
  store <2 x i64> <i64 1, i64 1>, <2 x i64>* %3854, align 16, !noalias !133
  call void @nnc_aten_adaptive_avg_pool2d(i64 2, i8** nonnull %.sub304.i, i64* nonnull %.sub305.i, i64* nonnull %.sub306.i, i8* nonnull %.sub307.i, i64 2, i64* nonnull %.sub308.i) #0, !noalias !136
  %3860 = alloca [3 x i8*], align 8
  %3861 = alloca [3 x i64], align 16
  %3862 = alloca [8 x i64], align 8
  %3863 = alloca [3 x i8], align 1
  %.sub312.i = getelementptr inbounds [3 x i8], [3 x i8]* %3863, i64 0, i64 0
  %.sub311.i = getelementptr inbounds [8 x i64], [8 x i64]* %3862, i64 0, i64 0
  %.sub310.i = getelementptr inbounds [3 x i64], [3 x i64]* %3861, i64 0, i64 0
  %.sub309.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %3860, i64 0, i64 0
  store i8* %malloccall40.i, i8** %.sub309.i, align 8, !noalias !133
  store i8 6, i8* %.sub312.i, align 1, !noalias !133
  %3864 = bitcast [8 x i64]* %3862 to <4 x i64>*
  store <4 x i64> <i64 1, i64 240, i64 1, i64 1>, <4 x i64>* %3864, align 8, !noalias !133
  %3865 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3860, i64 0, i64 1
  store i8* %malloccall51.i, i8** %3865, align 8, !noalias !133
  %3866 = getelementptr inbounds [3 x i8], [3 x i8]* %3863, i64 0, i64 1
  store i8 6, i8* %3866, align 1, !noalias !133
  %3867 = bitcast [3 x i64]* %3861 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %3867, align 16, !noalias !133
  %3868 = getelementptr inbounds [8 x i64], [8 x i64]* %3862, i64 0, i64 4
  %3869 = bitcast i64* %3868 to <4 x i64>*
  store <4 x i64> <i64 1, i64 960, i64 1, i64 1>, <4 x i64>* %3869, align 8, !noalias !133
  %3870 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3860, i64 0, i64 2
  %3871 = bitcast i8** %3870 to float**
  store float* %182, float** %3871, align 8, !noalias !133
  %3872 = getelementptr inbounds [3 x i8], [3 x i8]* %3863, i64 0, i64 2
  store i8 6, i8* %3872, align 1, !noalias !133
  %3873 = getelementptr inbounds [3 x i64], [3 x i64]* %3861, i64 0, i64 2
  store i64 0, i64* %3873, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub309.i, i64* nonnull %.sub310.i, i64* nonnull %.sub311.i, i8* nonnull %.sub312.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !137
  %3874 = alloca [3 x i8*], align 8
  %3875 = alloca [3 x i64], align 16
  %3876 = alloca [8 x i64], align 8
  %3877 = alloca [3 x i8], align 1
  %.sub317.i = getelementptr inbounds [3 x i8], [3 x i8]* %3877, i64 0, i64 0
  %.sub316.i = getelementptr inbounds [8 x i64], [8 x i64]* %3876, i64 0, i64 0
  %.sub315.i = getelementptr inbounds [3 x i64], [3 x i64]* %3875, i64 0, i64 0
  %.sub314.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %3874, i64 0, i64 0
  store i8* %malloccall55.i, i8** %.sub314.i, align 8, !noalias !133
  store i8 6, i8* %.sub317.i, align 1, !noalias !133
  %3878 = bitcast [8 x i64]* %3876 to <4 x i64>*
  store <4 x i64> <i64 1, i64 960, i64 1, i64 1>, <4 x i64>* %3878, align 8, !noalias !133
  %3879 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3874, i64 0, i64 1
  store i8* %malloccall40.i, i8** %3879, align 8, !noalias !133
  %3880 = getelementptr inbounds [3 x i8], [3 x i8]* %3877, i64 0, i64 1
  store i8 6, i8* %3880, align 1, !noalias !133
  %3881 = bitcast [3 x i64]* %3875 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %3881, align 16, !noalias !133
  %3882 = getelementptr inbounds [8 x i64], [8 x i64]* %3876, i64 0, i64 4
  %3883 = bitcast i64* %3882 to <4 x i64>*
  store <4 x i64> <i64 1, i64 240, i64 1, i64 1>, <4 x i64>* %3883, align 8, !noalias !133
  %3884 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3874, i64 0, i64 2
  %3885 = bitcast i8** %3884 to float**
  store float* %185, float** %3885, align 8, !noalias !133
  %3886 = getelementptr inbounds [3 x i8], [3 x i8]* %3877, i64 0, i64 2
  store i8 6, i8* %3886, align 1, !noalias !133
  %3887 = getelementptr inbounds [3 x i64], [3 x i64]* %3875, i64 0, i64 2
  store i64 0, i64* %3887, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub314.i, i64* nonnull %.sub315.i, i64* nonnull %.sub316.i, i8* nonnull %.sub317.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !138
  br label %cond546.preheader.i

cond537.preheader.i:                              ; preds = %cond537.preheader.i, %cond534.preheader.i
  %3888 = phi i64 [ 0, %cond534.preheader.i ], [ %3936, %cond537.preheader.i ]
  %3889 = mul nuw nsw i64 %3888, 7
  %3890 = add nuw nsw i64 %3889, %3848
  %3891 = getelementptr float, float* %279, i64 %3890
  %3892 = getelementptr float, float* %314, i64 %3890
  %3893 = bitcast float* %3891 to <4 x float>*
  %3894 = load <4 x float>, <4 x float>* %3893, align 4, !noalias !133
  %3895 = fadd <4 x float> %3894, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %3896 = fcmp olt <4 x float> %3895, zeroinitializer
  %3897 = select <4 x i1> %3896, <4 x float> zeroinitializer, <4 x float> %3895
  %3898 = fcmp ogt <4 x float> %3897, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %3899 = select <4 x i1> %3898, <4 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <4 x float> %3897
  %3900 = fmul <4 x float> %3894, %3899
  %3901 = fdiv <4 x float> %3900, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %3902 = bitcast float* %3892 to <4 x float>*
  store <4 x float> %3901, <4 x float>* %3902, align 4, !noalias !133
  %3903 = add nuw nsw i64 %3849, %3889
  %3904 = getelementptr float, float* %279, i64 %3903
  %3905 = load float, float* %3904, align 4, !noalias !133
  %3906 = fadd float %3905, 3.000000e+00
  %3907 = fcmp olt float %3906, 0.000000e+00
  %3908 = select i1 %3907, float 0.000000e+00, float %3906
  %3909 = fcmp ogt float %3908, 6.000000e+00
  %3910 = select i1 %3909, float 6.000000e+00, float %3908
  %3911 = fmul float %3905, %3910
  %3912 = fdiv float %3911, 6.000000e+00
  %3913 = getelementptr float, float* %314, i64 %3903
  store float %3912, float* %3913, align 4, !noalias !133
  %3914 = add nuw nsw i64 %3903, 1
  %3915 = getelementptr float, float* %279, i64 %3914
  %3916 = load float, float* %3915, align 4, !noalias !133
  %3917 = fadd float %3916, 3.000000e+00
  %3918 = fcmp olt float %3917, 0.000000e+00
  %3919 = select i1 %3918, float 0.000000e+00, float %3917
  %3920 = fcmp ogt float %3919, 6.000000e+00
  %3921 = select i1 %3920, float 6.000000e+00, float %3919
  %3922 = fmul float %3916, %3921
  %3923 = fdiv float %3922, 6.000000e+00
  %3924 = getelementptr float, float* %314, i64 %3914
  store float %3923, float* %3924, align 4, !noalias !133
  %3925 = add nuw nsw i64 %3903, 2
  %3926 = getelementptr float, float* %279, i64 %3925
  %3927 = load float, float* %3926, align 4, !noalias !133
  %3928 = fadd float %3927, 3.000000e+00
  %3929 = fcmp olt float %3928, 0.000000e+00
  %3930 = select i1 %3929, float 0.000000e+00, float %3928
  %3931 = fcmp ogt float %3930, 6.000000e+00
  %3932 = select i1 %3931, float 6.000000e+00, float %3930
  %3933 = fmul float %3927, %3932
  %3934 = fdiv float %3933, 6.000000e+00
  %3935 = getelementptr float, float* %314, i64 %3925
  store float %3934, float* %3935, align 4, !noalias !133
  %3936 = add nuw nsw i64 %3888, 1
  %exitcond441.not.i = icmp eq i64 %3936, 7
  br i1 %exitcond441.not.i, label %exit536.i, label %cond537.preheader.i

exit536.i:                                        ; preds = %cond537.preheader.i
  %3937 = add nuw nsw i64 %3847, 1
  %exitcond442.not.i = icmp eq i64 %3937, 960
  br i1 %exitcond442.not.i, label %exit533.i, label %cond534.preheader.i

cond546.preheader.i:                              ; preds = %cond546.preheader.i, %exit533.i
  %3938 = phi i64 [ 0, %exit533.i ], [ %3995, %cond546.preheader.i ]
  %3939 = mul nuw nsw i64 %3938, 49
  %3940 = getelementptr float, float* %288, i64 %3938
  %3941 = load float, float* %3940, align 4, !noalias !133
  %3942 = fadd float %3941, 3.000000e+00
  %3943 = fcmp olt float %3942, 0.000000e+00
  %3944 = select i1 %3943, float 0.000000e+00, float %3942
  %3945 = fcmp ogt float %3944, 6.000000e+00
  %.op380.i = fdiv float %3942, 6.000000e+00
  %.op379.i = select i1 %3943, float 0.000000e+00, float %.op380.i
  %3946 = select i1 %3945, float 1.000000e+00, float %.op379.i
  %3947 = getelementptr float, float* %314, i64 %3939
  %3948 = getelementptr float, float* %246, i64 %3939
  %3949 = bitcast float* %3947 to <8 x float>*
  %3950 = load <8 x float>, <8 x float>* %3949, align 4, !noalias !133
  %3951 = insertelement <8 x float> poison, float %3946, i32 0
  %3952 = shufflevector <8 x float> %3951, <8 x float> undef, <8 x i32> zeroinitializer
  %3953 = fmul <8 x float> %3950, %3952
  %3954 = bitcast float* %3948 to <8 x float>*
  store <8 x float> %3953, <8 x float>* %3954, align 4, !noalias !133
  %3955 = add nuw nsw i64 %3939, 8
  %3956 = getelementptr float, float* %314, i64 %3955
  %3957 = getelementptr float, float* %246, i64 %3955
  %3958 = bitcast float* %3956 to <8 x float>*
  %3959 = load <8 x float>, <8 x float>* %3958, align 4, !noalias !133
  %3960 = fmul <8 x float> %3959, %3952
  %3961 = bitcast float* %3957 to <8 x float>*
  store <8 x float> %3960, <8 x float>* %3961, align 4, !noalias !133
  %3962 = add nuw nsw i64 %3939, 16
  %3963 = getelementptr float, float* %314, i64 %3962
  %3964 = getelementptr float, float* %246, i64 %3962
  %3965 = bitcast float* %3963 to <8 x float>*
  %3966 = load <8 x float>, <8 x float>* %3965, align 4, !noalias !133
  %3967 = fmul <8 x float> %3966, %3952
  %3968 = bitcast float* %3964 to <8 x float>*
  store <8 x float> %3967, <8 x float>* %3968, align 4, !noalias !133
  %3969 = add nuw nsw i64 %3939, 24
  %3970 = getelementptr float, float* %314, i64 %3969
  %3971 = getelementptr float, float* %246, i64 %3969
  %3972 = bitcast float* %3970 to <8 x float>*
  %3973 = load <8 x float>, <8 x float>* %3972, align 4, !noalias !133
  %3974 = fmul <8 x float> %3973, %3952
  %3975 = bitcast float* %3971 to <8 x float>*
  store <8 x float> %3974, <8 x float>* %3975, align 4, !noalias !133
  %3976 = add nuw nsw i64 %3939, 32
  %3977 = getelementptr float, float* %314, i64 %3976
  %3978 = getelementptr float, float* %246, i64 %3976
  %3979 = bitcast float* %3977 to <8 x float>*
  %3980 = load <8 x float>, <8 x float>* %3979, align 4, !noalias !133
  %3981 = fmul <8 x float> %3980, %3952
  %3982 = bitcast float* %3978 to <8 x float>*
  store <8 x float> %3981, <8 x float>* %3982, align 4, !noalias !133
  %3983 = add nuw nsw i64 %3939, 40
  %3984 = getelementptr float, float* %314, i64 %3983
  %3985 = getelementptr float, float* %246, i64 %3983
  %3986 = bitcast float* %3984 to <8 x float>*
  %3987 = load <8 x float>, <8 x float>* %3986, align 4, !noalias !133
  %3988 = fmul <8 x float> %3987, %3952
  %3989 = bitcast float* %3985 to <8 x float>*
  store <8 x float> %3988, <8 x float>* %3989, align 4, !noalias !133
  %3990 = add nuw nsw i64 %3939, 48
  %3991 = getelementptr float, float* %314, i64 %3990
  %3992 = load float, float* %3991, align 4, !noalias !133
  %3993 = fmul float %3992, %3946
  %3994 = getelementptr float, float* %246, i64 %3990
  store float %3993, float* %3994, align 4, !noalias !133
  %3995 = add nuw nsw i64 %3938, 1
  %exitcond438.not.i = icmp eq i64 %3995, 960
  br i1 %exitcond438.not.i, label %exit545.i, label %cond546.preheader.i

exit545.i:                                        ; preds = %cond546.preheader.i
  %3996 = alloca [3 x i8*], align 8
  %3997 = alloca [3 x i64], align 16
  %3998 = alloca [8 x i64], align 8
  %3999 = alloca [3 x i8], align 1
  %.sub322.i = getelementptr inbounds [3 x i8], [3 x i8]* %3999, i64 0, i64 0
  %.sub321.i = getelementptr inbounds [8 x i64], [8 x i64]* %3998, i64 0, i64 0
  %.sub320.i = getelementptr inbounds [3 x i64], [3 x i64]* %3997, i64 0, i64 0
  %.sub319.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %3996, i64 0, i64 0
  store i8* %malloccall85.i, i8** %.sub319.i, align 8, !noalias !133
  store i8 6, i8* %.sub322.i, align 1, !noalias !133
  %4000 = bitcast [8 x i64]* %3998 to <4 x i64>*
  store <4 x i64> <i64 1, i64 160, i64 7, i64 7>, <4 x i64>* %4000, align 8, !noalias !133
  %4001 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3996, i64 0, i64 1
  store i8* %malloccall4.i, i8** %4001, align 8, !noalias !133
  %4002 = getelementptr inbounds [3 x i8], [3 x i8]* %3999, i64 0, i64 1
  store i8 6, i8* %4002, align 1, !noalias !133
  %4003 = bitcast [3 x i64]* %3997 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %4003, align 16, !noalias !133
  %4004 = getelementptr inbounds [8 x i64], [8 x i64]* %3998, i64 0, i64 4
  %4005 = bitcast i64* %4004 to <4 x i64>*
  store <4 x i64> <i64 1, i64 960, i64 7, i64 7>, <4 x i64>* %4005, align 8, !noalias !133
  %4006 = getelementptr inbounds [3 x i8*], [3 x i8*]* %3996, i64 0, i64 2
  %4007 = bitcast i8** %4006 to float**
  store float* %188, float** %4007, align 8, !noalias !133
  %4008 = getelementptr inbounds [3 x i8], [3 x i8]* %3999, i64 0, i64 2
  store i8 6, i8* %4008, align 1, !noalias !133
  %4009 = getelementptr inbounds [3 x i64], [3 x i64]* %3997, i64 0, i64 2
  store i64 0, i64* %4009, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub319.i, i64* nonnull %.sub320.i, i64* nonnull %.sub321.i, i8* nonnull %.sub322.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !139
  br label %cond558.preheader.i

cond558.preheader.i:                              ; preds = %cond558.preheader.i, %exit545.i
  %4010 = phi i64 [ 0, %exit545.i ], [ %4078, %cond558.preheader.i ]
  %4011 = mul nuw nsw i64 %4010, 49
  %4012 = getelementptr float, float* %310, i64 %4011
  %4013 = getelementptr float, float* %308, i64 %4011
  %4014 = getelementptr float, float* %244, i64 %4011
  %4015 = bitcast float* %4012 to <8 x float>*
  %4016 = load <8 x float>, <8 x float>* %4015, align 4, !noalias !133
  %4017 = bitcast float* %4013 to <8 x float>*
  %4018 = load <8 x float>, <8 x float>* %4017, align 4, !noalias !133
  %4019 = fadd <8 x float> %4016, %4018
  %4020 = bitcast float* %4014 to <8 x float>*
  store <8 x float> %4019, <8 x float>* %4020, align 4, !noalias !133
  %4021 = add nuw nsw i64 %4011, 8
  %4022 = getelementptr float, float* %310, i64 %4021
  %4023 = getelementptr float, float* %308, i64 %4021
  %4024 = getelementptr float, float* %244, i64 %4021
  %4025 = bitcast float* %4022 to <8 x float>*
  %4026 = load <8 x float>, <8 x float>* %4025, align 4, !noalias !133
  %4027 = bitcast float* %4023 to <8 x float>*
  %4028 = load <8 x float>, <8 x float>* %4027, align 4, !noalias !133
  %4029 = fadd <8 x float> %4026, %4028
  %4030 = bitcast float* %4024 to <8 x float>*
  store <8 x float> %4029, <8 x float>* %4030, align 4, !noalias !133
  %4031 = add nuw nsw i64 %4011, 16
  %4032 = getelementptr float, float* %310, i64 %4031
  %4033 = getelementptr float, float* %308, i64 %4031
  %4034 = getelementptr float, float* %244, i64 %4031
  %4035 = bitcast float* %4032 to <8 x float>*
  %4036 = load <8 x float>, <8 x float>* %4035, align 4, !noalias !133
  %4037 = bitcast float* %4033 to <8 x float>*
  %4038 = load <8 x float>, <8 x float>* %4037, align 4, !noalias !133
  %4039 = fadd <8 x float> %4036, %4038
  %4040 = bitcast float* %4034 to <8 x float>*
  store <8 x float> %4039, <8 x float>* %4040, align 4, !noalias !133
  %4041 = add nuw nsw i64 %4011, 24
  %4042 = getelementptr float, float* %310, i64 %4041
  %4043 = getelementptr float, float* %308, i64 %4041
  %4044 = getelementptr float, float* %244, i64 %4041
  %4045 = bitcast float* %4042 to <8 x float>*
  %4046 = load <8 x float>, <8 x float>* %4045, align 4, !noalias !133
  %4047 = bitcast float* %4043 to <8 x float>*
  %4048 = load <8 x float>, <8 x float>* %4047, align 4, !noalias !133
  %4049 = fadd <8 x float> %4046, %4048
  %4050 = bitcast float* %4044 to <8 x float>*
  store <8 x float> %4049, <8 x float>* %4050, align 4, !noalias !133
  %4051 = add nuw nsw i64 %4011, 32
  %4052 = getelementptr float, float* %310, i64 %4051
  %4053 = getelementptr float, float* %308, i64 %4051
  %4054 = getelementptr float, float* %244, i64 %4051
  %4055 = bitcast float* %4052 to <8 x float>*
  %4056 = load <8 x float>, <8 x float>* %4055, align 4, !noalias !133
  %4057 = bitcast float* %4053 to <8 x float>*
  %4058 = load <8 x float>, <8 x float>* %4057, align 4, !noalias !133
  %4059 = fadd <8 x float> %4056, %4058
  %4060 = bitcast float* %4054 to <8 x float>*
  store <8 x float> %4059, <8 x float>* %4060, align 4, !noalias !133
  %4061 = add nuw nsw i64 %4011, 40
  %4062 = getelementptr float, float* %310, i64 %4061
  %4063 = getelementptr float, float* %308, i64 %4061
  %4064 = getelementptr float, float* %244, i64 %4061
  %4065 = bitcast float* %4062 to <8 x float>*
  %4066 = load <8 x float>, <8 x float>* %4065, align 4, !noalias !133
  %4067 = bitcast float* %4063 to <8 x float>*
  %4068 = load <8 x float>, <8 x float>* %4067, align 4, !noalias !133
  %4069 = fadd <8 x float> %4066, %4068
  %4070 = bitcast float* %4064 to <8 x float>*
  store <8 x float> %4069, <8 x float>* %4070, align 4, !noalias !133
  %4071 = add nuw nsw i64 %4011, 48
  %4072 = getelementptr float, float* %310, i64 %4071
  %4073 = load float, float* %4072, align 4, !noalias !133
  %4074 = getelementptr float, float* %308, i64 %4071
  %4075 = load float, float* %4074, align 4, !noalias !133
  %4076 = fadd float %4073, %4075
  %4077 = getelementptr float, float* %244, i64 %4071
  store float %4076, float* %4077, align 4, !noalias !133
  %4078 = add nuw nsw i64 %4010, 1
  %exitcond434.not.i = icmp eq i64 %4078, 160
  br i1 %exitcond434.not.i, label %exit557.i, label %cond558.preheader.i

exit557.i:                                        ; preds = %cond558.preheader.i
  %4079 = alloca [3 x i8*], align 8
  %4080 = alloca [3 x i64], align 16
  %4081 = alloca [8 x i64], align 8
  %4082 = alloca [3 x i8], align 1
  %.sub327.i = getelementptr inbounds [3 x i8], [3 x i8]* %4082, i64 0, i64 0
  %.sub326.i = getelementptr inbounds [8 x i64], [8 x i64]* %4081, i64 0, i64 0
  %.sub325.i = getelementptr inbounds [3 x i64], [3 x i64]* %4080, i64 0, i64 0
  %.sub324.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %4079, i64 0, i64 0
  store i8* %malloccall.i, i8** %.sub324.i, align 8, !noalias !133
  store i8 6, i8* %.sub327.i, align 1, !noalias !133
  %4083 = bitcast [8 x i64]* %4081 to <4 x i64>*
  store <4 x i64> <i64 1, i64 960, i64 7, i64 7>, <4 x i64>* %4083, align 8, !noalias !133
  %4084 = getelementptr inbounds [3 x i8*], [3 x i8*]* %4079, i64 0, i64 1
  store i8* %malloccall1.i, i8** %4084, align 8, !noalias !133
  %4085 = getelementptr inbounds [3 x i8], [3 x i8]* %4082, i64 0, i64 1
  store i8 6, i8* %4085, align 1, !noalias !133
  %4086 = bitcast [3 x i64]* %4080 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %4086, align 16, !noalias !133
  %4087 = getelementptr inbounds [8 x i64], [8 x i64]* %4081, i64 0, i64 4
  %4088 = bitcast i64* %4087 to <4 x i64>*
  store <4 x i64> <i64 1, i64 160, i64 7, i64 7>, <4 x i64>* %4088, align 8, !noalias !133
  %4089 = getelementptr inbounds [3 x i8*], [3 x i8*]* %4079, i64 0, i64 2
  %4090 = bitcast i8** %4089 to float**
  store float* %191, float** %4090, align 8, !noalias !133
  %4091 = getelementptr inbounds [3 x i8], [3 x i8]* %4082, i64 0, i64 2
  store i8 6, i8* %4091, align 1, !noalias !133
  %4092 = getelementptr inbounds [3 x i64], [3 x i64]* %4080, i64 0, i64 2
  store i64 0, i64* %4092, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub324.i, i64* nonnull %.sub325.i, i64* nonnull %.sub326.i, i8* nonnull %.sub327.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !140
  br label %cond570.preheader.i

cond570.preheader.i:                              ; preds = %exit572.i, %exit557.i
  %4093 = phi i64 [ 0, %exit557.i ], [ %4159, %exit572.i ]
  %4094 = mul nuw nsw i64 %4093, 49
  %4095 = add nuw nsw i64 %4094, 4
  br label %cond573.preheader.i

exit569.i:                                        ; preds = %exit572.i
  %4096 = alloca [3 x i8*], align 8
  %4097 = alloca [3 x i64], align 16
  %4098 = alloca [8 x i64], align 8
  %4099 = alloca [3 x i8], align 1
  %.sub332.i = getelementptr inbounds [3 x i8], [3 x i8]* %4099, i64 0, i64 0
  %.sub331.i = getelementptr inbounds [8 x i64], [8 x i64]* %4098, i64 0, i64 0
  %.sub330.i = getelementptr inbounds [3 x i64], [3 x i64]* %4097, i64 0, i64 0
  %.sub329.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %4096, i64 0, i64 0
  store i8* %malloccall68.i, i8** %.sub329.i, align 8, !noalias !133
  store i8 6, i8* %.sub332.i, align 1, !noalias !133
  %4100 = bitcast [8 x i64]* %4098 to <4 x i64>*
  store <4 x i64> <i64 1, i64 960, i64 7, i64 7>, <4 x i64>* %4100, align 8, !noalias !133
  %4101 = getelementptr inbounds [3 x i8*], [3 x i8*]* %4096, i64 0, i64 1
  store i8* %malloccall84.i, i8** %4101, align 8, !noalias !133
  %4102 = getelementptr inbounds [3 x i8], [3 x i8]* %4099, i64 0, i64 1
  store i8 6, i8* %4102, align 1, !noalias !133
  %4103 = bitcast [3 x i64]* %4097 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %4103, align 16, !noalias !133
  %4104 = getelementptr inbounds [8 x i64], [8 x i64]* %4098, i64 0, i64 4
  %4105 = bitcast i64* %4104 to <4 x i64>*
  store <4 x i64> <i64 1, i64 960, i64 7, i64 7>, <4 x i64>* %4105, align 8, !noalias !133
  %4106 = getelementptr inbounds [3 x i8*], [3 x i8*]* %4096, i64 0, i64 2
  %4107 = bitcast i8** %4106 to float**
  store float* %194, float** %4107, align 8, !noalias !133
  %4108 = getelementptr inbounds [3 x i8], [3 x i8]* %4099, i64 0, i64 2
  store i8 6, i8* %4108, align 1, !noalias !133
  %4109 = getelementptr inbounds [3 x i64], [3 x i64]* %4097, i64 0, i64 2
  store i64 0, i64* %4109, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub329.i, i64* nonnull %.sub330.i, i64* nonnull %.sub331.i, i8* nonnull %.sub332.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !141
  br label %cond582.preheader.i

cond573.preheader.i:                              ; preds = %cond573.preheader.i, %cond570.preheader.i
  %4110 = phi i64 [ 0, %cond570.preheader.i ], [ %4158, %cond573.preheader.i ]
  %4111 = mul nuw nsw i64 %4110, 7
  %4112 = add nuw nsw i64 %4111, %4094
  %4113 = getelementptr float, float* %243, i64 %4112
  %4114 = getelementptr float, float* %309, i64 %4112
  %4115 = bitcast float* %4113 to <4 x float>*
  %4116 = load <4 x float>, <4 x float>* %4115, align 4, !noalias !133
  %4117 = fadd <4 x float> %4116, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %4118 = fcmp olt <4 x float> %4117, zeroinitializer
  %4119 = select <4 x i1> %4118, <4 x float> zeroinitializer, <4 x float> %4117
  %4120 = fcmp ogt <4 x float> %4119, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4121 = select <4 x i1> %4120, <4 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <4 x float> %4119
  %4122 = fmul <4 x float> %4116, %4121
  %4123 = fdiv <4 x float> %4122, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4124 = bitcast float* %4114 to <4 x float>*
  store <4 x float> %4123, <4 x float>* %4124, align 4, !noalias !133
  %4125 = add nuw nsw i64 %4095, %4111
  %4126 = getelementptr float, float* %243, i64 %4125
  %4127 = load float, float* %4126, align 4, !noalias !133
  %4128 = fadd float %4127, 3.000000e+00
  %4129 = fcmp olt float %4128, 0.000000e+00
  %4130 = select i1 %4129, float 0.000000e+00, float %4128
  %4131 = fcmp ogt float %4130, 6.000000e+00
  %4132 = select i1 %4131, float 6.000000e+00, float %4130
  %4133 = fmul float %4127, %4132
  %4134 = fdiv float %4133, 6.000000e+00
  %4135 = getelementptr float, float* %309, i64 %4125
  store float %4134, float* %4135, align 4, !noalias !133
  %4136 = add nuw nsw i64 %4125, 1
  %4137 = getelementptr float, float* %243, i64 %4136
  %4138 = load float, float* %4137, align 4, !noalias !133
  %4139 = fadd float %4138, 3.000000e+00
  %4140 = fcmp olt float %4139, 0.000000e+00
  %4141 = select i1 %4140, float 0.000000e+00, float %4139
  %4142 = fcmp ogt float %4141, 6.000000e+00
  %4143 = select i1 %4142, float 6.000000e+00, float %4141
  %4144 = fmul float %4138, %4143
  %4145 = fdiv float %4144, 6.000000e+00
  %4146 = getelementptr float, float* %309, i64 %4136
  store float %4145, float* %4146, align 4, !noalias !133
  %4147 = add nuw nsw i64 %4125, 2
  %4148 = getelementptr float, float* %243, i64 %4147
  %4149 = load float, float* %4148, align 4, !noalias !133
  %4150 = fadd float %4149, 3.000000e+00
  %4151 = fcmp olt float %4150, 0.000000e+00
  %4152 = select i1 %4151, float 0.000000e+00, float %4150
  %4153 = fcmp ogt float %4152, 6.000000e+00
  %4154 = select i1 %4153, float 6.000000e+00, float %4152
  %4155 = fmul float %4149, %4154
  %4156 = fdiv float %4155, 6.000000e+00
  %4157 = getelementptr float, float* %309, i64 %4147
  store float %4156, float* %4157, align 4, !noalias !133
  %4158 = add nuw nsw i64 %4110, 1
  %exitcond429.not.i = icmp eq i64 %4158, 7
  br i1 %exitcond429.not.i, label %exit572.i, label %cond573.preheader.i

exit572.i:                                        ; preds = %cond573.preheader.i
  %4159 = add nuw nsw i64 %4093, 1
  %exitcond430.not.i = icmp eq i64 %4159, 960
  br i1 %exitcond430.not.i, label %exit569.i, label %cond570.preheader.i

cond582.preheader.i:                              ; preds = %exit584.i, %exit569.i
  %4160 = phi i64 [ 0, %exit569.i ], [ %4250, %exit584.i ]
  %4161 = mul nuw nsw i64 %4160, 49
  %4162 = add nuw nsw i64 %4161, 4
  br label %cond585.preheader.i

exit581.i:                                        ; preds = %exit584.i
  %4163 = alloca [2 x i8*], align 8
  %4164 = alloca <2 x i64>, align 16
  %4165 = alloca [8 x i64], align 8
  %4166 = alloca [2 x i8], align 1
  %4167 = alloca <2 x i64>, align 16
  %.sub338.i = getelementptr inbounds <2 x i64>, <2 x i64>* %4167, i64 0, i64 0
  %.sub337.i = getelementptr inbounds [2 x i8], [2 x i8]* %4166, i64 0, i64 0
  %.sub336.i = getelementptr inbounds [8 x i64], [8 x i64]* %4165, i64 0, i64 0
  %.sub335.i = getelementptr inbounds <2 x i64>, <2 x i64>* %4164, i64 0, i64 0
  %.sub334.i = getelementptr inbounds [2 x i8*], [2 x i8*]* %4163, i64 0, i64 0
  store i8* %malloccall3.i, i8** %.sub334.i, align 8, !noalias !133
  store i8 6, i8* %.sub337.i, align 1, !noalias !133
  %4168 = bitcast [8 x i64]* %4165 to <4 x i64>*
  store <4 x i64> <i64 1, i64 960, i64 1, i64 1>, <4 x i64>* %4168, align 8, !noalias !133
  %4169 = getelementptr inbounds [2 x i8*], [2 x i8*]* %4163, i64 0, i64 1
  store i8* %malloccall91.i, i8** %4169, align 8, !noalias !133
  %4170 = getelementptr inbounds [2 x i8], [2 x i8]* %4166, i64 0, i64 1
  store i8 6, i8* %4170, align 1, !noalias !133
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %4164, align 16, !noalias !133
  %4171 = getelementptr inbounds [8 x i64], [8 x i64]* %4165, i64 0, i64 4
  %4172 = bitcast i64* %4171 to <4 x i64>*
  store <4 x i64> <i64 1, i64 960, i64 7, i64 7>, <4 x i64>* %4172, align 8, !noalias !133
  store <2 x i64> <i64 1, i64 1>, <2 x i64>* %4167, align 16, !noalias !133
  call void @nnc_aten_adaptive_avg_pool2d(i64 2, i8** nonnull %.sub334.i, i64* nonnull %.sub335.i, i64* nonnull %.sub336.i, i8* nonnull %.sub337.i, i64 2, i64* nonnull %.sub338.i) #0, !noalias !141
  %4173 = alloca [3 x i8*], align 8
  %4174 = alloca [3 x i64], align 16
  %4175 = alloca [8 x i64], align 8
  %4176 = alloca [3 x i8], align 1
  %.sub342.i = getelementptr inbounds [3 x i8], [3 x i8]* %4176, i64 0, i64 0
  %.sub341.i = getelementptr inbounds [8 x i64], [8 x i64]* %4175, i64 0, i64 0
  %.sub340.i = getelementptr inbounds [3 x i64], [3 x i64]* %4174, i64 0, i64 0
  %.sub339.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %4173, i64 0, i64 0
  store i8* %malloccall92.i, i8** %.sub339.i, align 8, !noalias !133
  store i8 6, i8* %.sub342.i, align 1, !noalias !133
  %4177 = bitcast [8 x i64]* %4175 to <4 x i64>*
  store <4 x i64> <i64 1, i64 240, i64 1, i64 1>, <4 x i64>* %4177, align 8, !noalias !133
  %4178 = getelementptr inbounds [3 x i8*], [3 x i8*]* %4173, i64 0, i64 1
  store i8* %malloccall3.i, i8** %4178, align 8, !noalias !133
  %4179 = getelementptr inbounds [3 x i8], [3 x i8]* %4176, i64 0, i64 1
  store i8 6, i8* %4179, align 1, !noalias !133
  %4180 = bitcast [3 x i64]* %4174 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %4180, align 16, !noalias !133
  %4181 = getelementptr inbounds [8 x i64], [8 x i64]* %4175, i64 0, i64 4
  %4182 = bitcast i64* %4181 to <4 x i64>*
  store <4 x i64> <i64 1, i64 960, i64 1, i64 1>, <4 x i64>* %4182, align 8, !noalias !133
  %4183 = getelementptr inbounds [3 x i8*], [3 x i8*]* %4173, i64 0, i64 2
  %4184 = bitcast i8** %4183 to float**
  store float* %197, float** %4184, align 8, !noalias !133
  %4185 = getelementptr inbounds [3 x i8], [3 x i8]* %4176, i64 0, i64 2
  store i8 6, i8* %4185, align 1, !noalias !133
  %4186 = getelementptr inbounds [3 x i64], [3 x i64]* %4174, i64 0, i64 2
  store i64 0, i64* %4186, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub339.i, i64* nonnull %.sub340.i, i64* nonnull %.sub341.i, i8* nonnull %.sub342.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !142
  %4187 = alloca [3 x i8*], align 8
  %4188 = alloca [3 x i64], align 16
  %4189 = alloca [8 x i64], align 8
  %4190 = alloca [3 x i8], align 1
  %.sub347.i = getelementptr inbounds [3 x i8], [3 x i8]* %4190, i64 0, i64 0
  %.sub346.i = getelementptr inbounds [8 x i64], [8 x i64]* %4189, i64 0, i64 0
  %.sub345.i = getelementptr inbounds [3 x i64], [3 x i64]* %4188, i64 0, i64 0
  %.sub344.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %4187, i64 0, i64 0
  store i8* %malloccall93.i, i8** %.sub344.i, align 8, !noalias !133
  store i8 6, i8* %.sub347.i, align 1, !noalias !133
  %4191 = bitcast [8 x i64]* %4189 to <4 x i64>*
  store <4 x i64> <i64 1, i64 960, i64 1, i64 1>, <4 x i64>* %4191, align 8, !noalias !133
  %4192 = getelementptr inbounds [3 x i8*], [3 x i8*]* %4187, i64 0, i64 1
  store i8* %malloccall92.i, i8** %4192, align 8, !noalias !133
  %4193 = getelementptr inbounds [3 x i8], [3 x i8]* %4190, i64 0, i64 1
  store i8 6, i8* %4193, align 1, !noalias !133
  %4194 = bitcast [3 x i64]* %4188 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %4194, align 16, !noalias !133
  %4195 = getelementptr inbounds [8 x i64], [8 x i64]* %4189, i64 0, i64 4
  %4196 = bitcast i64* %4195 to <4 x i64>*
  store <4 x i64> <i64 1, i64 240, i64 1, i64 1>, <4 x i64>* %4196, align 8, !noalias !133
  %4197 = getelementptr inbounds [3 x i8*], [3 x i8*]* %4187, i64 0, i64 2
  %4198 = bitcast i8** %4197 to float**
  store float* %200, float** %4198, align 8, !noalias !133
  %4199 = getelementptr inbounds [3 x i8], [3 x i8]* %4190, i64 0, i64 2
  store i8 6, i8* %4199, align 1, !noalias !133
  %4200 = getelementptr inbounds [3 x i64], [3 x i64]* %4188, i64 0, i64 2
  store i64 0, i64* %4200, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub344.i, i64* nonnull %.sub345.i, i64* nonnull %.sub346.i, i8* nonnull %.sub347.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !143
  br label %cond594.preheader.i

cond585.preheader.i:                              ; preds = %cond585.preheader.i, %cond582.preheader.i
  %4201 = phi i64 [ 0, %cond582.preheader.i ], [ %4249, %cond585.preheader.i ]
  %4202 = mul nuw nsw i64 %4201, 7
  %4203 = add nuw nsw i64 %4202, %4161
  %4204 = getelementptr float, float* %297, i64 %4203
  %4205 = getelementptr float, float* %316, i64 %4203
  %4206 = bitcast float* %4204 to <4 x float>*
  %4207 = load <4 x float>, <4 x float>* %4206, align 4, !noalias !133
  %4208 = fadd <4 x float> %4207, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %4209 = fcmp olt <4 x float> %4208, zeroinitializer
  %4210 = select <4 x i1> %4209, <4 x float> zeroinitializer, <4 x float> %4208
  %4211 = fcmp ogt <4 x float> %4210, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4212 = select <4 x i1> %4211, <4 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <4 x float> %4210
  %4213 = fmul <4 x float> %4207, %4212
  %4214 = fdiv <4 x float> %4213, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4215 = bitcast float* %4205 to <4 x float>*
  store <4 x float> %4214, <4 x float>* %4215, align 4, !noalias !133
  %4216 = add nuw nsw i64 %4162, %4202
  %4217 = getelementptr float, float* %297, i64 %4216
  %4218 = load float, float* %4217, align 4, !noalias !133
  %4219 = fadd float %4218, 3.000000e+00
  %4220 = fcmp olt float %4219, 0.000000e+00
  %4221 = select i1 %4220, float 0.000000e+00, float %4219
  %4222 = fcmp ogt float %4221, 6.000000e+00
  %4223 = select i1 %4222, float 6.000000e+00, float %4221
  %4224 = fmul float %4218, %4223
  %4225 = fdiv float %4224, 6.000000e+00
  %4226 = getelementptr float, float* %316, i64 %4216
  store float %4225, float* %4226, align 4, !noalias !133
  %4227 = add nuw nsw i64 %4216, 1
  %4228 = getelementptr float, float* %297, i64 %4227
  %4229 = load float, float* %4228, align 4, !noalias !133
  %4230 = fadd float %4229, 3.000000e+00
  %4231 = fcmp olt float %4230, 0.000000e+00
  %4232 = select i1 %4231, float 0.000000e+00, float %4230
  %4233 = fcmp ogt float %4232, 6.000000e+00
  %4234 = select i1 %4233, float 6.000000e+00, float %4232
  %4235 = fmul float %4229, %4234
  %4236 = fdiv float %4235, 6.000000e+00
  %4237 = getelementptr float, float* %316, i64 %4227
  store float %4236, float* %4237, align 4, !noalias !133
  %4238 = add nuw nsw i64 %4216, 2
  %4239 = getelementptr float, float* %297, i64 %4238
  %4240 = load float, float* %4239, align 4, !noalias !133
  %4241 = fadd float %4240, 3.000000e+00
  %4242 = fcmp olt float %4241, 0.000000e+00
  %4243 = select i1 %4242, float 0.000000e+00, float %4241
  %4244 = fcmp ogt float %4243, 6.000000e+00
  %4245 = select i1 %4244, float 6.000000e+00, float %4243
  %4246 = fmul float %4240, %4245
  %4247 = fdiv float %4246, 6.000000e+00
  %4248 = getelementptr float, float* %316, i64 %4238
  store float %4247, float* %4248, align 4, !noalias !133
  %4249 = add nuw nsw i64 %4201, 1
  %exitcond425.not.i = icmp eq i64 %4249, 7
  br i1 %exitcond425.not.i, label %exit584.i, label %cond585.preheader.i

exit584.i:                                        ; preds = %cond585.preheader.i
  %4250 = add nuw nsw i64 %4160, 1
  %exitcond426.not.i = icmp eq i64 %4250, 960
  br i1 %exitcond426.not.i, label %exit581.i, label %cond582.preheader.i

cond594.preheader.i:                              ; preds = %cond594.preheader.i, %exit581.i
  %4251 = phi i64 [ 0, %exit581.i ], [ %4308, %cond594.preheader.i ]
  %4252 = mul nuw nsw i64 %4251, 49
  %4253 = getelementptr float, float* %317, i64 %4251
  %4254 = load float, float* %4253, align 4, !noalias !133
  %4255 = fadd float %4254, 3.000000e+00
  %4256 = fcmp olt float %4255, 0.000000e+00
  %4257 = select i1 %4256, float 0.000000e+00, float %4255
  %4258 = fcmp ogt float %4257, 6.000000e+00
  %.op376.i = fdiv float %4255, 6.000000e+00
  %.op375.i = select i1 %4256, float 0.000000e+00, float %.op376.i
  %4259 = select i1 %4258, float 1.000000e+00, float %.op375.i
  %4260 = getelementptr float, float* %316, i64 %4252
  %4261 = getelementptr float, float* %285, i64 %4252
  %4262 = bitcast float* %4260 to <8 x float>*
  %4263 = load <8 x float>, <8 x float>* %4262, align 4, !noalias !133
  %4264 = insertelement <8 x float> poison, float %4259, i32 0
  %4265 = shufflevector <8 x float> %4264, <8 x float> undef, <8 x i32> zeroinitializer
  %4266 = fmul <8 x float> %4263, %4265
  %4267 = bitcast float* %4261 to <8 x float>*
  store <8 x float> %4266, <8 x float>* %4267, align 4, !noalias !133
  %4268 = add nuw nsw i64 %4252, 8
  %4269 = getelementptr float, float* %316, i64 %4268
  %4270 = getelementptr float, float* %285, i64 %4268
  %4271 = bitcast float* %4269 to <8 x float>*
  %4272 = load <8 x float>, <8 x float>* %4271, align 4, !noalias !133
  %4273 = fmul <8 x float> %4272, %4265
  %4274 = bitcast float* %4270 to <8 x float>*
  store <8 x float> %4273, <8 x float>* %4274, align 4, !noalias !133
  %4275 = add nuw nsw i64 %4252, 16
  %4276 = getelementptr float, float* %316, i64 %4275
  %4277 = getelementptr float, float* %285, i64 %4275
  %4278 = bitcast float* %4276 to <8 x float>*
  %4279 = load <8 x float>, <8 x float>* %4278, align 4, !noalias !133
  %4280 = fmul <8 x float> %4279, %4265
  %4281 = bitcast float* %4277 to <8 x float>*
  store <8 x float> %4280, <8 x float>* %4281, align 4, !noalias !133
  %4282 = add nuw nsw i64 %4252, 24
  %4283 = getelementptr float, float* %316, i64 %4282
  %4284 = getelementptr float, float* %285, i64 %4282
  %4285 = bitcast float* %4283 to <8 x float>*
  %4286 = load <8 x float>, <8 x float>* %4285, align 4, !noalias !133
  %4287 = fmul <8 x float> %4286, %4265
  %4288 = bitcast float* %4284 to <8 x float>*
  store <8 x float> %4287, <8 x float>* %4288, align 4, !noalias !133
  %4289 = add nuw nsw i64 %4252, 32
  %4290 = getelementptr float, float* %316, i64 %4289
  %4291 = getelementptr float, float* %285, i64 %4289
  %4292 = bitcast float* %4290 to <8 x float>*
  %4293 = load <8 x float>, <8 x float>* %4292, align 4, !noalias !133
  %4294 = fmul <8 x float> %4293, %4265
  %4295 = bitcast float* %4291 to <8 x float>*
  store <8 x float> %4294, <8 x float>* %4295, align 4, !noalias !133
  %4296 = add nuw nsw i64 %4252, 40
  %4297 = getelementptr float, float* %316, i64 %4296
  %4298 = getelementptr float, float* %285, i64 %4296
  %4299 = bitcast float* %4297 to <8 x float>*
  %4300 = load <8 x float>, <8 x float>* %4299, align 4, !noalias !133
  %4301 = fmul <8 x float> %4300, %4265
  %4302 = bitcast float* %4298 to <8 x float>*
  store <8 x float> %4301, <8 x float>* %4302, align 4, !noalias !133
  %4303 = add nuw nsw i64 %4252, 48
  %4304 = getelementptr float, float* %316, i64 %4303
  %4305 = load float, float* %4304, align 4, !noalias !133
  %4306 = fmul float %4305, %4259
  %4307 = getelementptr float, float* %285, i64 %4303
  store float %4306, float* %4307, align 4, !noalias !133
  %4308 = add nuw nsw i64 %4251, 1
  %exitcond422.not.i = icmp eq i64 %4308, 960
  br i1 %exitcond422.not.i, label %exit593.i, label %cond594.preheader.i

exit593.i:                                        ; preds = %cond594.preheader.i
  %4309 = alloca [3 x i8*], align 8
  %4310 = alloca [3 x i64], align 16
  %4311 = alloca [8 x i64], align 8
  %4312 = alloca [3 x i8], align 1
  %.sub352.i = getelementptr inbounds [3 x i8], [3 x i8]* %4312, i64 0, i64 0
  %.sub351.i = getelementptr inbounds [8 x i64], [8 x i64]* %4311, i64 0, i64 0
  %.sub350.i = getelementptr inbounds [3 x i64], [3 x i64]* %4310, i64 0, i64 0
  %.sub349.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %4309, i64 0, i64 0
  store i8* %malloccall94.i, i8** %.sub349.i, align 8, !noalias !133
  store i8 6, i8* %.sub352.i, align 1, !noalias !133
  %4313 = bitcast [8 x i64]* %4311 to <4 x i64>*
  store <4 x i64> <i64 1, i64 160, i64 7, i64 7>, <4 x i64>* %4313, align 8, !noalias !133
  %4314 = getelementptr inbounds [3 x i8*], [3 x i8*]* %4309, i64 0, i64 1
  store i8* %malloccall50.i, i8** %4314, align 8, !noalias !133
  %4315 = getelementptr inbounds [3 x i8], [3 x i8]* %4312, i64 0, i64 1
  store i8 6, i8* %4315, align 1, !noalias !133
  %4316 = bitcast [3 x i64]* %4310 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %4316, align 16, !noalias !133
  %4317 = getelementptr inbounds [8 x i64], [8 x i64]* %4311, i64 0, i64 4
  %4318 = bitcast i64* %4317 to <4 x i64>*
  store <4 x i64> <i64 1, i64 960, i64 7, i64 7>, <4 x i64>* %4318, align 8, !noalias !133
  %4319 = getelementptr inbounds [3 x i8*], [3 x i8*]* %4309, i64 0, i64 2
  %4320 = bitcast i8** %4319 to float**
  store float* %203, float** %4320, align 8, !noalias !133
  %4321 = getelementptr inbounds [3 x i8], [3 x i8]* %4312, i64 0, i64 2
  store i8 6, i8* %4321, align 1, !noalias !133
  %4322 = getelementptr inbounds [3 x i64], [3 x i64]* %4310, i64 0, i64 2
  store i64 0, i64* %4322, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub349.i, i64* nonnull %.sub350.i, i64* nonnull %.sub351.i, i8* nonnull %.sub352.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !144
  br label %cond606.preheader.i

cond606.preheader.i:                              ; preds = %cond606.preheader.i, %exit593.i
  %4323 = phi i64 [ 0, %exit593.i ], [ %4391, %cond606.preheader.i ]
  %4324 = mul nuw nsw i64 %4323, 49
  %4325 = getelementptr float, float* %318, i64 %4324
  %4326 = getelementptr float, float* %244, i64 %4324
  %4327 = getelementptr float, float* %319, i64 %4324
  %4328 = bitcast float* %4325 to <8 x float>*
  %4329 = load <8 x float>, <8 x float>* %4328, align 4, !noalias !133
  %4330 = bitcast float* %4326 to <8 x float>*
  %4331 = load <8 x float>, <8 x float>* %4330, align 4, !noalias !133
  %4332 = fadd <8 x float> %4329, %4331
  %4333 = bitcast float* %4327 to <8 x float>*
  store <8 x float> %4332, <8 x float>* %4333, align 4, !noalias !133
  %4334 = add nuw nsw i64 %4324, 8
  %4335 = getelementptr float, float* %318, i64 %4334
  %4336 = getelementptr float, float* %244, i64 %4334
  %4337 = getelementptr float, float* %319, i64 %4334
  %4338 = bitcast float* %4335 to <8 x float>*
  %4339 = load <8 x float>, <8 x float>* %4338, align 4, !noalias !133
  %4340 = bitcast float* %4336 to <8 x float>*
  %4341 = load <8 x float>, <8 x float>* %4340, align 4, !noalias !133
  %4342 = fadd <8 x float> %4339, %4341
  %4343 = bitcast float* %4337 to <8 x float>*
  store <8 x float> %4342, <8 x float>* %4343, align 4, !noalias !133
  %4344 = add nuw nsw i64 %4324, 16
  %4345 = getelementptr float, float* %318, i64 %4344
  %4346 = getelementptr float, float* %244, i64 %4344
  %4347 = getelementptr float, float* %319, i64 %4344
  %4348 = bitcast float* %4345 to <8 x float>*
  %4349 = load <8 x float>, <8 x float>* %4348, align 4, !noalias !133
  %4350 = bitcast float* %4346 to <8 x float>*
  %4351 = load <8 x float>, <8 x float>* %4350, align 4, !noalias !133
  %4352 = fadd <8 x float> %4349, %4351
  %4353 = bitcast float* %4347 to <8 x float>*
  store <8 x float> %4352, <8 x float>* %4353, align 4, !noalias !133
  %4354 = add nuw nsw i64 %4324, 24
  %4355 = getelementptr float, float* %318, i64 %4354
  %4356 = getelementptr float, float* %244, i64 %4354
  %4357 = getelementptr float, float* %319, i64 %4354
  %4358 = bitcast float* %4355 to <8 x float>*
  %4359 = load <8 x float>, <8 x float>* %4358, align 4, !noalias !133
  %4360 = bitcast float* %4356 to <8 x float>*
  %4361 = load <8 x float>, <8 x float>* %4360, align 4, !noalias !133
  %4362 = fadd <8 x float> %4359, %4361
  %4363 = bitcast float* %4357 to <8 x float>*
  store <8 x float> %4362, <8 x float>* %4363, align 4, !noalias !133
  %4364 = add nuw nsw i64 %4324, 32
  %4365 = getelementptr float, float* %318, i64 %4364
  %4366 = getelementptr float, float* %244, i64 %4364
  %4367 = getelementptr float, float* %319, i64 %4364
  %4368 = bitcast float* %4365 to <8 x float>*
  %4369 = load <8 x float>, <8 x float>* %4368, align 4, !noalias !133
  %4370 = bitcast float* %4366 to <8 x float>*
  %4371 = load <8 x float>, <8 x float>* %4370, align 4, !noalias !133
  %4372 = fadd <8 x float> %4369, %4371
  %4373 = bitcast float* %4367 to <8 x float>*
  store <8 x float> %4372, <8 x float>* %4373, align 4, !noalias !133
  %4374 = add nuw nsw i64 %4324, 40
  %4375 = getelementptr float, float* %318, i64 %4374
  %4376 = getelementptr float, float* %244, i64 %4374
  %4377 = getelementptr float, float* %319, i64 %4374
  %4378 = bitcast float* %4375 to <8 x float>*
  %4379 = load <8 x float>, <8 x float>* %4378, align 4, !noalias !133
  %4380 = bitcast float* %4376 to <8 x float>*
  %4381 = load <8 x float>, <8 x float>* %4380, align 4, !noalias !133
  %4382 = fadd <8 x float> %4379, %4381
  %4383 = bitcast float* %4377 to <8 x float>*
  store <8 x float> %4382, <8 x float>* %4383, align 4, !noalias !133
  %4384 = add nuw nsw i64 %4324, 48
  %4385 = getelementptr float, float* %318, i64 %4384
  %4386 = load float, float* %4385, align 4, !noalias !133
  %4387 = getelementptr float, float* %244, i64 %4384
  %4388 = load float, float* %4387, align 4, !noalias !133
  %4389 = fadd float %4386, %4388
  %4390 = getelementptr float, float* %319, i64 %4384
  store float %4389, float* %4390, align 4, !noalias !133
  %4391 = add nuw nsw i64 %4323, 1
  %exitcond418.not.i = icmp eq i64 %4391, 160
  br i1 %exitcond418.not.i, label %exit605.i, label %cond606.preheader.i

exit605.i:                                        ; preds = %cond606.preheader.i
  %4392 = alloca [3 x i8*], align 8
  %4393 = alloca [3 x i64], align 16
  %4394 = alloca [8 x i64], align 8
  %4395 = alloca [3 x i8], align 1
  %.sub357.i = getelementptr inbounds [3 x i8], [3 x i8]* %4395, i64 0, i64 0
  %.sub356.i = getelementptr inbounds [8 x i64], [8 x i64]* %4394, i64 0, i64 0
  %.sub355.i = getelementptr inbounds [3 x i64], [3 x i64]* %4393, i64 0, i64 0
  %.sub354.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %4392, i64 0, i64 0
  store i8* %malloccall96.i, i8** %.sub354.i, align 8, !noalias !133
  store i8 6, i8* %.sub357.i, align 1, !noalias !133
  %4396 = bitcast [8 x i64]* %4394 to <4 x i64>*
  store <4 x i64> <i64 1, i64 960, i64 7, i64 7>, <4 x i64>* %4396, align 8, !noalias !133
  %4397 = getelementptr inbounds [3 x i8*], [3 x i8*]* %4392, i64 0, i64 1
  store i8* %malloccall95.i, i8** %4397, align 8, !noalias !133
  %4398 = getelementptr inbounds [3 x i8], [3 x i8]* %4395, i64 0, i64 1
  store i8 6, i8* %4398, align 1, !noalias !133
  %4399 = bitcast [3 x i64]* %4393 to <2 x i64>*
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %4399, align 16, !noalias !133
  %4400 = getelementptr inbounds [8 x i64], [8 x i64]* %4394, i64 0, i64 4
  %4401 = bitcast i64* %4400 to <4 x i64>*
  store <4 x i64> <i64 1, i64 160, i64 7, i64 7>, <4 x i64>* %4401, align 8, !noalias !133
  %4402 = getelementptr inbounds [3 x i8*], [3 x i8*]* %4392, i64 0, i64 2
  %4403 = bitcast i8** %4402 to float**
  store float* %206, float** %4403, align 8, !noalias !133
  %4404 = getelementptr inbounds [3 x i8], [3 x i8]* %4395, i64 0, i64 2
  store i8 6, i8* %4404, align 1, !noalias !133
  %4405 = getelementptr inbounds [3 x i64], [3 x i64]* %4393, i64 0, i64 2
  store i64 0, i64* %4405, align 16, !noalias !133
  call void @nnc_prepacked_conv2d_clamp_run(i64 3, i8** nonnull %.sub354.i, i64* nonnull %.sub355.i, i64* nonnull %.sub356.i, i8* nonnull %.sub357.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !145
  br label %cond618.preheader.i

cond618.preheader.i:                              ; preds = %exit620.i, %exit605.i
  %4406 = phi i64 [ 0, %exit605.i ], [ %4468, %exit620.i ]
  %4407 = mul nuw nsw i64 %4406, 49
  %4408 = add nuw nsw i64 %4407, 4
  br label %cond621.preheader.i

exit617.i:                                        ; preds = %exit620.i
  %4409 = alloca [2 x i8*], align 8
  %4410 = alloca <2 x i64>, align 16
  %4411 = alloca [8 x i64], align 8
  %4412 = alloca [2 x i8], align 1
  %4413 = alloca <2 x i64>, align 16
  %.sub363.i = getelementptr inbounds <2 x i64>, <2 x i64>* %4413, i64 0, i64 0
  %.sub362.i = getelementptr inbounds [2 x i8], [2 x i8]* %4412, i64 0, i64 0
  %.sub361.i = getelementptr inbounds [8 x i64], [8 x i64]* %4411, i64 0, i64 0
  %.sub360.i = getelementptr inbounds <2 x i64>, <2 x i64>* %4410, i64 0, i64 0
  %.sub359.i = getelementptr inbounds [2 x i8*], [2 x i8*]* %4409, i64 0, i64 0
  store i8* %malloccall98.i, i8** %.sub359.i, align 8, !noalias !133
  store i8 6, i8* %.sub362.i, align 1, !noalias !133
  %4414 = bitcast [8 x i64]* %4411 to <4 x i64>*
  store <4 x i64> <i64 1, i64 960, i64 1, i64 1>, <4 x i64>* %4414, align 8, !noalias !133
  %4415 = getelementptr inbounds [2 x i8*], [2 x i8*]* %4409, i64 0, i64 1
  store i8* %malloccall97.i, i8** %4415, align 8, !noalias !133
  %4416 = getelementptr inbounds [2 x i8], [2 x i8]* %4412, i64 0, i64 1
  store i8 6, i8* %4416, align 1, !noalias !133
  store <2 x i64> <i64 4, i64 4>, <2 x i64>* %4410, align 16, !noalias !133
  %4417 = getelementptr inbounds [8 x i64], [8 x i64]* %4411, i64 0, i64 4
  %4418 = bitcast i64* %4417 to <4 x i64>*
  store <4 x i64> <i64 1, i64 960, i64 7, i64 7>, <4 x i64>* %4418, align 8, !noalias !133
  store <2 x i64> <i64 1, i64 1>, <2 x i64>* %4413, align 16, !noalias !133
  call void @nnc_aten_adaptive_avg_pool2d(i64 2, i8** nonnull %.sub359.i, i64* nonnull %.sub360.i, i64* nonnull %.sub361.i, i8* nonnull %.sub362.i, i64 2, i64* nonnull %.sub363.i) #0, !noalias !145
  br label %cond630.preheader.i

cond621.preheader.i:                              ; preds = %cond621.preheader.i, %cond618.preheader.i
  %4419 = phi i64 [ 0, %cond618.preheader.i ], [ %4467, %cond621.preheader.i ]
  %4420 = mul nuw nsw i64 %4419, 7
  %4421 = add nuw nsw i64 %4420, %4407
  %4422 = getelementptr float, float* %320, i64 %4421
  %4423 = getelementptr float, float* %321, i64 %4421
  %4424 = bitcast float* %4422 to <4 x float>*
  %4425 = load <4 x float>, <4 x float>* %4424, align 4, !noalias !133
  %4426 = fadd <4 x float> %4425, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %4427 = fcmp olt <4 x float> %4426, zeroinitializer
  %4428 = select <4 x i1> %4427, <4 x float> zeroinitializer, <4 x float> %4426
  %4429 = fcmp ogt <4 x float> %4428, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4430 = select <4 x i1> %4429, <4 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <4 x float> %4428
  %4431 = fmul <4 x float> %4425, %4430
  %4432 = fdiv <4 x float> %4431, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4433 = bitcast float* %4423 to <4 x float>*
  store <4 x float> %4432, <4 x float>* %4433, align 4, !noalias !133
  %4434 = add nuw nsw i64 %4408, %4420
  %4435 = getelementptr float, float* %320, i64 %4434
  %4436 = load float, float* %4435, align 4, !noalias !133
  %4437 = fadd float %4436, 3.000000e+00
  %4438 = fcmp olt float %4437, 0.000000e+00
  %4439 = select i1 %4438, float 0.000000e+00, float %4437
  %4440 = fcmp ogt float %4439, 6.000000e+00
  %4441 = select i1 %4440, float 6.000000e+00, float %4439
  %4442 = fmul float %4436, %4441
  %4443 = fdiv float %4442, 6.000000e+00
  %4444 = getelementptr float, float* %321, i64 %4434
  store float %4443, float* %4444, align 4, !noalias !133
  %4445 = add nuw nsw i64 %4434, 1
  %4446 = getelementptr float, float* %320, i64 %4445
  %4447 = load float, float* %4446, align 4, !noalias !133
  %4448 = fadd float %4447, 3.000000e+00
  %4449 = fcmp olt float %4448, 0.000000e+00
  %4450 = select i1 %4449, float 0.000000e+00, float %4448
  %4451 = fcmp ogt float %4450, 6.000000e+00
  %4452 = select i1 %4451, float 6.000000e+00, float %4450
  %4453 = fmul float %4447, %4452
  %4454 = fdiv float %4453, 6.000000e+00
  %4455 = getelementptr float, float* %321, i64 %4445
  store float %4454, float* %4455, align 4, !noalias !133
  %4456 = add nuw nsw i64 %4434, 2
  %4457 = getelementptr float, float* %320, i64 %4456
  %4458 = load float, float* %4457, align 4, !noalias !133
  %4459 = fadd float %4458, 3.000000e+00
  %4460 = fcmp olt float %4459, 0.000000e+00
  %4461 = select i1 %4460, float 0.000000e+00, float %4459
  %4462 = fcmp ogt float %4461, 6.000000e+00
  %4463 = select i1 %4462, float 6.000000e+00, float %4461
  %4464 = fmul float %4458, %4463
  %4465 = fdiv float %4464, 6.000000e+00
  %4466 = getelementptr float, float* %321, i64 %4456
  store float %4465, float* %4466, align 4, !noalias !133
  %4467 = add nuw nsw i64 %4419, 1
  %exitcond413.not.i = icmp eq i64 %4467, 7
  br i1 %exitcond413.not.i, label %exit620.i, label %cond621.preheader.i

exit620.i:                                        ; preds = %cond621.preheader.i
  %4468 = add nuw nsw i64 %4406, 1
  %exitcond414.not.i = icmp eq i64 %4468, 960
  br i1 %exitcond414.not.i, label %exit617.i, label %cond618.preheader.i

cond630.preheader.i:                              ; preds = %cond630.preheader.i, %exit617.i
  %4469 = phi i64 [ 0, %exit617.i ], [ %4478, %cond630.preheader.i ]
  %4470 = shl i64 %4469, 5
  %scevgep.i = getelementptr i8, i8* %malloccall99.i, i64 %4470
  %scevgep409.i = getelementptr i8, i8* %malloccall98.i, i64 %4470
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 4 dereferenceable(32) %scevgep.i, i8* nonnull align 4 dereferenceable(32) %scevgep409.i, i64 32, i1 false) #0, !noalias !133
  %4471 = or i64 %4470, 32
  %scevgep.1.i = getelementptr i8, i8* %malloccall99.i, i64 %4471
  %scevgep409.1.i = getelementptr i8, i8* %malloccall98.i, i64 %4471
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 4 dereferenceable(32) %scevgep.1.i, i8* nonnull align 4 dereferenceable(32) %scevgep409.1.i, i64 32, i1 false) #0, !noalias !133
  %4472 = or i64 %4470, 64
  %scevgep.2.i = getelementptr i8, i8* %malloccall99.i, i64 %4472
  %scevgep409.2.i = getelementptr i8, i8* %malloccall98.i, i64 %4472
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 4 dereferenceable(32) %scevgep.2.i, i8* nonnull align 4 dereferenceable(32) %scevgep409.2.i, i64 32, i1 false) #0, !noalias !133
  %4473 = or i64 %4470, 96
  %scevgep.3.i = getelementptr i8, i8* %malloccall99.i, i64 %4473
  %scevgep409.3.i = getelementptr i8, i8* %malloccall98.i, i64 %4473
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 4 dereferenceable(32) %scevgep.3.i, i8* nonnull align 4 dereferenceable(32) %scevgep409.3.i, i64 32, i1 false) #0, !noalias !133
  %4474 = or i64 %4470, 128
  %scevgep.4.i = getelementptr i8, i8* %malloccall99.i, i64 %4474
  %scevgep409.4.i = getelementptr i8, i8* %malloccall98.i, i64 %4474
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 4 dereferenceable(32) %scevgep.4.i, i8* nonnull align 4 dereferenceable(32) %scevgep409.4.i, i64 32, i1 false) #0, !noalias !133
  %4475 = or i64 %4470, 160
  %scevgep.5.i = getelementptr i8, i8* %malloccall99.i, i64 %4475
  %scevgep409.5.i = getelementptr i8, i8* %malloccall98.i, i64 %4475
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 4 dereferenceable(32) %scevgep.5.i, i8* nonnull align 4 dereferenceable(32) %scevgep409.5.i, i64 32, i1 false) #0, !noalias !133
  %4476 = or i64 %4470, 192
  %scevgep.6.i = getelementptr i8, i8* %malloccall99.i, i64 %4476
  %scevgep409.6.i = getelementptr i8, i8* %malloccall98.i, i64 %4476
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 4 dereferenceable(32) %scevgep.6.i, i8* nonnull align 4 dereferenceable(32) %scevgep409.6.i, i64 32, i1 false) #0, !noalias !133
  %4477 = or i64 %4470, 224
  %scevgep.7.i = getelementptr i8, i8* %malloccall99.i, i64 %4477
  %scevgep409.7.i = getelementptr i8, i8* %malloccall98.i, i64 %4477
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 4 dereferenceable(32) %scevgep.7.i, i8* nonnull align 4 dereferenceable(32) %scevgep409.7.i, i64 32, i1 false) #0, !noalias !133
  %4478 = add nuw nsw i64 %4469, 8
  %exitcond410.not.7.i = icmp eq i64 %4478, 120
  br i1 %exitcond410.not.7.i, label %exit629.i, label %cond630.preheader.i

exit629.i:                                        ; preds = %cond630.preheader.i
  %4479 = alloca [3 x i8*], align 8
  %4480 = alloca [3 x i64], align 16
  %4481 = alloca <4 x i64>, align 8
  %4482 = alloca [3 x i8], align 1
  %.sub367.i = getelementptr inbounds [3 x i8], [3 x i8]* %4482, i64 0, i64 0
  %.sub366.i = getelementptr inbounds <4 x i64>, <4 x i64>* %4481, i64 0, i64 0
  %.sub365.i = getelementptr inbounds [3 x i64], [3 x i64]* %4480, i64 0, i64 0
  %.sub364.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %4479, i64 0, i64 0
  store i8* %malloccall100.i, i8** %.sub364.i, align 8, !noalias !133
  store i8 6, i8* %.sub367.i, align 1, !noalias !133
  %4483 = getelementptr inbounds [3 x i8*], [3 x i8*]* %4479, i64 0, i64 1
  store i8* %malloccall99.i, i8** %4483, align 8, !noalias !133
  %4484 = getelementptr inbounds [3 x i8], [3 x i8]* %4482, i64 0, i64 1
  store i8 6, i8* %4484, align 1, !noalias !133
  %4485 = bitcast [3 x i64]* %4480 to <2 x i64>*
  store <2 x i64> <i64 2, i64 2>, <2 x i64>* %4485, align 16, !noalias !133
  store <4 x i64> <i64 1, i64 1280, i64 1, i64 960>, <4 x i64>* %4481, align 8, !noalias !133
  %4486 = getelementptr inbounds [3 x i8*], [3 x i8*]* %4479, i64 0, i64 2
  %4487 = bitcast i8** %4486 to float**
  store float* %209, float** %4487, align 8, !noalias !133
  %4488 = getelementptr inbounds [3 x i8], [3 x i8]* %4482, i64 0, i64 2
  store i8 6, i8* %4488, align 1, !noalias !133
  %4489 = getelementptr inbounds [3 x i64], [3 x i64]* %4480, i64 0, i64 2
  store i64 0, i64* %4489, align 16, !noalias !133
  call void @nnc_prepacked_linear_clamp_run(i64 3, i8** nonnull %.sub364.i, i64* nonnull %.sub365.i, i64* nonnull %.sub366.i, i8* nonnull %.sub367.i, i64 0, i64* nonnull %.sub14.i) #0, !noalias !146
  %4490 = getelementptr i8, i8* %malloccall101.i, i64 -28
  %4491 = bitcast i8* %4490 to float*
  br label %vector.body.i

vector.body.i:                                    ; preds = %vector.body.i, %exit629.i
  %index.i = phi i64 [ 0, %exit629.i ], [ %index.next.i, %vector.body.i ]
  %4492 = shl i64 %index.i, 3
  %4493 = getelementptr float, float* %322, i64 %4492
  %4494 = bitcast float* %4493 to <64 x float>*
  %wide.vec.i = load <64 x float>, <64 x float>* %4494, align 4, !noalias !133
  %strided.vec.i = shufflevector <64 x float> %wide.vec.i, <64 x float> poison, <8 x i32> <i32 0, i32 8, i32 16, i32 24, i32 32, i32 40, i32 48, i32 56>
  %strided.vec569.i = shufflevector <64 x float> %wide.vec.i, <64 x float> poison, <8 x i32> <i32 1, i32 9, i32 17, i32 25, i32 33, i32 41, i32 49, i32 57>
  %strided.vec570.i = shufflevector <64 x float> %wide.vec.i, <64 x float> poison, <8 x i32> <i32 2, i32 10, i32 18, i32 26, i32 34, i32 42, i32 50, i32 58>
  %strided.vec571.i = shufflevector <64 x float> %wide.vec.i, <64 x float> poison, <8 x i32> <i32 3, i32 11, i32 19, i32 27, i32 35, i32 43, i32 51, i32 59>
  %strided.vec572.i = shufflevector <64 x float> %wide.vec.i, <64 x float> poison, <8 x i32> <i32 4, i32 12, i32 20, i32 28, i32 36, i32 44, i32 52, i32 60>
  %strided.vec573.i = shufflevector <64 x float> %wide.vec.i, <64 x float> poison, <8 x i32> <i32 5, i32 13, i32 21, i32 29, i32 37, i32 45, i32 53, i32 61>
  %strided.vec574.i = shufflevector <64 x float> %wide.vec.i, <64 x float> poison, <8 x i32> <i32 6, i32 14, i32 22, i32 30, i32 38, i32 46, i32 54, i32 62>
  %strided.vec575.i = shufflevector <64 x float> %wide.vec.i, <64 x float> poison, <8 x i32> <i32 7, i32 15, i32 23, i32 31, i32 39, i32 47, i32 55, i32 63>
  %4495 = fadd <8 x float> %strided.vec.i, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %4496 = fcmp olt <8 x float> %4495, zeroinitializer
  %4497 = select <8 x i1> %4496, <8 x float> zeroinitializer, <8 x float> %4495
  %4498 = fcmp ogt <8 x float> %4497, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4499 = select <8 x i1> %4498, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %4497
  %4500 = fmul <8 x float> %strided.vec.i, %4499
  %4501 = fdiv <8 x float> %4500, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4502 = fadd <8 x float> %strided.vec569.i, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %4503 = fcmp olt <8 x float> %4502, zeroinitializer
  %4504 = select <8 x i1> %4503, <8 x float> zeroinitializer, <8 x float> %4502
  %4505 = fcmp ogt <8 x float> %4504, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4506 = select <8 x i1> %4505, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %4504
  %4507 = fmul <8 x float> %strided.vec569.i, %4506
  %4508 = fdiv <8 x float> %4507, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4509 = fadd <8 x float> %strided.vec570.i, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %4510 = fcmp olt <8 x float> %4509, zeroinitializer
  %4511 = select <8 x i1> %4510, <8 x float> zeroinitializer, <8 x float> %4509
  %4512 = fcmp ogt <8 x float> %4511, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4513 = select <8 x i1> %4512, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %4511
  %4514 = fmul <8 x float> %strided.vec570.i, %4513
  %4515 = fdiv <8 x float> %4514, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4516 = fadd <8 x float> %strided.vec571.i, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %4517 = fcmp olt <8 x float> %4516, zeroinitializer
  %4518 = select <8 x i1> %4517, <8 x float> zeroinitializer, <8 x float> %4516
  %4519 = fcmp ogt <8 x float> %4518, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4520 = select <8 x i1> %4519, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %4518
  %4521 = fmul <8 x float> %strided.vec571.i, %4520
  %4522 = fdiv <8 x float> %4521, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4523 = fadd <8 x float> %strided.vec572.i, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %4524 = fcmp olt <8 x float> %4523, zeroinitializer
  %4525 = select <8 x i1> %4524, <8 x float> zeroinitializer, <8 x float> %4523
  %4526 = fcmp ogt <8 x float> %4525, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4527 = select <8 x i1> %4526, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %4525
  %4528 = fmul <8 x float> %strided.vec572.i, %4527
  %4529 = fdiv <8 x float> %4528, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4530 = fadd <8 x float> %strided.vec573.i, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %4531 = fcmp olt <8 x float> %4530, zeroinitializer
  %4532 = select <8 x i1> %4531, <8 x float> zeroinitializer, <8 x float> %4530
  %4533 = fcmp ogt <8 x float> %4532, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4534 = select <8 x i1> %4533, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %4532
  %4535 = fmul <8 x float> %strided.vec573.i, %4534
  %4536 = fdiv <8 x float> %4535, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4537 = fadd <8 x float> %strided.vec574.i, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %4538 = fcmp olt <8 x float> %4537, zeroinitializer
  %4539 = select <8 x i1> %4538, <8 x float> zeroinitializer, <8 x float> %4537
  %4540 = fcmp ogt <8 x float> %4539, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4541 = select <8 x i1> %4540, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %4539
  %4542 = fmul <8 x float> %strided.vec574.i, %4541
  %4543 = fdiv <8 x float> %4542, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4544 = or i64 %4492, 7
  %4545 = fadd <8 x float> %strided.vec575.i, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %4546 = fcmp olt <8 x float> %4545, zeroinitializer
  %4547 = select <8 x i1> %4546, <8 x float> zeroinitializer, <8 x float> %4545
  %4548 = fcmp ogt <8 x float> %4547, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4549 = select <8 x i1> %4548, <8 x float> <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>, <8 x float> %4547
  %4550 = fmul <8 x float> %strided.vec575.i, %4549
  %4551 = fdiv <8 x float> %4550, <float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00, float 6.000000e+00>
  %4552 = getelementptr float, float* %4491, i64 %4544
  %4553 = bitcast float* %4552 to <64 x float>*
  %4554 = shufflevector <8 x float> %4501, <8 x float> %4508, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %4555 = shufflevector <8 x float> %4515, <8 x float> %4522, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %4556 = shufflevector <8 x float> %4529, <8 x float> %4536, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %4557 = shufflevector <8 x float> %4543, <8 x float> %4551, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %4558 = shufflevector <16 x float> %4554, <16 x float> %4555, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %4559 = shufflevector <16 x float> %4556, <16 x float> %4557, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %interleaved.vec.i = shufflevector <32 x float> %4558, <32 x float> %4559, <64 x i32> <i32 0, i32 8, i32 16, i32 24, i32 32, i32 40, i32 48, i32 56, i32 1, i32 9, i32 17, i32 25, i32 33, i32 41, i32 49, i32 57, i32 2, i32 10, i32 18, i32 26, i32 34, i32 42, i32 50, i32 58, i32 3, i32 11, i32 19, i32 27, i32 35, i32 43, i32 51, i32 59, i32 4, i32 12, i32 20, i32 28, i32 36, i32 44, i32 52, i32 60, i32 5, i32 13, i32 21, i32 29, i32 37, i32 45, i32 53, i32 61, i32 6, i32 14, i32 22, i32 30, i32 38, i32 46, i32 54, i32 62, i32 7, i32 15, i32 23, i32 31, i32 39, i32 47, i32 55, i32 63>
  store <64 x float> %interleaved.vec.i, <64 x float>* %4553, align 4, !noalias !133
  %index.next.i = add i64 %index.i, 8
  %4560 = icmp eq i64 %index.next.i, 160
  br i1 %4560, label %pytorch.exit, label %vector.body.i, !llvm.loop !147

pytorch.exit:                                     ; preds = %vector.body.i
  %4561 = alloca [3 x i8*], align 8
  %4562 = alloca [3 x i64], align 16
  %4563 = alloca <4 x i64>, align 8
  %4564 = alloca [3 x i8], align 1
  %.sub372.i = getelementptr inbounds [3 x i8], [3 x i8]* %4564, i64 0, i64 0
  %.sub371.i = getelementptr inbounds <4 x i64>, <4 x i64>* %4563, i64 0, i64 0
  %.sub370.i = getelementptr inbounds [3 x i64], [3 x i64]* %4562, i64 0, i64 0
  %.sub369.i = getelementptr inbounds [3 x i8*], [3 x i8*]* %4561, i64 0, i64 0
  %4565 = bitcast [3 x i8*]* %4561 to float**
  store float* %20, float** %4565, align 8, !noalias !133
  store i8 6, i8* %.sub372.i, align 1, !noalias !133
  %4566 = getelementptr inbounds [3 x i8*], [3 x i8*]* %4561, i64 0, i64 1
  store i8* %malloccall101.i, i8** %4566, align 8, !noalias !133
  %4567 = getelementptr inbounds [3 x i8], [3 x i8]* %4564, i64 0, i64 1
  store i8 6, i8* %4567, align 1, !noalias !133
  %4568 = bitcast [3 x i64]* %4562 to <2 x i64>*
  store <2 x i64> <i64 2, i64 2>, <2 x i64>* %4568, align 16, !noalias !133
  store <4 x i64> <i64 1, i64 1000, i64 1, i64 1280>, <4 x i64>* %4563, align 8, !noalias !133
  %4569 = getelementptr inbounds [3 x i8*], [3 x i8*]* %4561, i64 0, i64 2
  %4570 = bitcast i8** %4569 to float**
  store float* %212, float** %4570, align 8, !noalias !133
  %4571 = getelementptr inbounds [3 x i8], [3 x i8]* %4564, i64 0, i64 2
  store i8 6, i8* %4571, align 1, !noalias !133
  %4572 = getelementptr inbounds [3 x i64], [3 x i64]* %4562, i64 0, i64 2
  store i64 0, i64* %4572, align 16, !noalias !133
  call void @nnc_prepacked_linear_clamp_run(i64 3, i8** nonnull %.sub369.i, i64* nonnull %.sub370.i, i64* nonnull %.sub371.i, i8* nonnull %.sub372.i, i64 0, i64* nonnull %.sub14.i) #0
  tail call void @free(i8* nonnull %malloccall101.i) #0
  tail call void @free(i8* nonnull %malloccall100.i) #0
  tail call void @free(i8* %malloccall99.i) #0
  tail call void @free(i8* %malloccall98.i) #0
  tail call void @free(i8* %malloccall97.i) #0
  tail call void @free(i8* %malloccall96.i) #0
  tail call void @free(i8* %malloccall95.i) #0
  tail call void @free(i8* %malloccall94.i) #0
  tail call void @free(i8* %malloccall93.i) #0
  tail call void @free(i8* %malloccall92.i) #0
  tail call void @free(i8* %malloccall91.i) #0
  tail call void @free(i8* %malloccall90.i) #0
  tail call void @free(i8* %malloccall89.i) #0
  tail call void @free(i8* %malloccall88.i) #0
  tail call void @free(i8* %malloccall87.i) #0
  tail call void @free(i8* %malloccall86.i) #0
  tail call void @free(i8* %malloccall85.i) #0
  tail call void @free(i8* %malloccall84.i) #0
  tail call void @free(i8* %malloccall83.i) #0
  tail call void @free(i8* %malloccall82.i) #0
  tail call void @free(i8* %malloccall81.i) #0
  tail call void @free(i8* %malloccall80.i) #0
  tail call void @free(i8* %malloccall79.i) #0
  tail call void @free(i8* %malloccall78.i) #0
  tail call void @free(i8* %malloccall77.i) #0
  tail call void @free(i8* %malloccall76.i) #0
  tail call void @free(i8* %malloccall75.i) #0
  tail call void @free(i8* %malloccall74.i) #0
  tail call void @free(i8* %malloccall73.i) #0
  tail call void @free(i8* %malloccall72.i) #0
  tail call void @free(i8* %malloccall71.i) #0
  tail call void @free(i8* %malloccall70.i) #0
  tail call void @free(i8* %malloccall69.i) #0
  tail call void @free(i8* %malloccall68.i) #0
  tail call void @free(i8* %malloccall67.i) #0
  tail call void @free(i8* %malloccall66.i) #0
  tail call void @free(i8* %malloccall65.i) #0
  tail call void @free(i8* %malloccall64.i) #0
  tail call void @free(i8* %malloccall63.i) #0
  tail call void @free(i8* %malloccall62.i) #0
  tail call void @free(i8* %malloccall61.i) #0
  tail call void @free(i8* %malloccall60.i) #0
  tail call void @free(i8* %malloccall59.i) #0
  tail call void @free(i8* %malloccall58.i) #0
  tail call void @free(i8* %malloccall57.i) #0
  tail call void @free(i8* %malloccall56.i) #0
  tail call void @free(i8* %malloccall55.i) #0
  tail call void @free(i8* %malloccall54.i) #0
  tail call void @free(i8* %malloccall53.i) #0
  tail call void @free(i8* %malloccall52.i) #0
  tail call void @free(i8* %malloccall51.i) #0
  tail call void @free(i8* %malloccall50.i) #0
  tail call void @free(i8* %malloccall49.i) #0
  tail call void @free(i8* %malloccall48.i) #0
  tail call void @free(i8* %malloccall47.i) #0
  tail call void @free(i8* %malloccall46.i) #0
  tail call void @free(i8* %malloccall45.i) #0
  tail call void @free(i8* %malloccall44.i) #0
  tail call void @free(i8* %malloccall43.i) #0
  tail call void @free(i8* %malloccall42.i) #0
  tail call void @free(i8* %malloccall41.i) #0
  tail call void @free(i8* %malloccall40.i) #0
  tail call void @free(i8* %malloccall39.i) #0
  tail call void @free(i8* %malloccall38.i) #0
  tail call void @free(i8* %malloccall37.i) #0
  tail call void @free(i8* %malloccall36.i) #0
  tail call void @free(i8* %malloccall35.i) #0
  tail call void @free(i8* %malloccall34.i) #0
  tail call void @free(i8* %malloccall33.i) #0
  tail call void @free(i8* %malloccall32.i) #0
  tail call void @free(i8* %malloccall31.i) #0
  tail call void @free(i8* %malloccall30.i) #0
  tail call void @free(i8* %malloccall29.i) #0
  tail call void @free(i8* %malloccall28.i) #0
  tail call void @free(i8* %malloccall27.i) #0
  tail call void @free(i8* %malloccall26.i) #0
  tail call void @free(i8* %malloccall25.i) #0
  tail call void @free(i8* %malloccall24.i) #0
  tail call void @free(i8* %malloccall23.i) #0
  tail call void @free(i8* %malloccall22.i) #0
  tail call void @free(i8* %malloccall21.i) #0
  tail call void @free(i8* %malloccall20.i) #0
  tail call void @free(i8* %malloccall19.i) #0
  tail call void @free(i8* %malloccall18.i) #0
  tail call void @free(i8* %malloccall17.i) #0
  tail call void @free(i8* %malloccall16.i) #0
  tail call void @free(i8* %malloccall15.i) #0
  tail call void @free(i8* %malloccall14.i) #0
  tail call void @free(i8* %malloccall13.i) #0
  tail call void @free(i8* %malloccall12.i) #0
  tail call void @free(i8* %malloccall11.i) #0
  tail call void @free(i8* %malloccall10.i) #0
  tail call void @free(i8* %malloccall9.i) #0
  tail call void @free(i8* %malloccall8.i) #0
  tail call void @free(i8* %malloccall7.i) #0
  tail call void @free(i8* %malloccall6.i) #0
  tail call void @free(i8* %malloccall5.i) #0
  tail call void @free(i8* %malloccall4.i) #0
  tail call void @free(i8* %malloccall3.i) #0
  tail call void @free(i8* %malloccall2.i) #0
  tail call void @free(i8* %malloccall1.i) #0
  tail call void @free(i8* %malloccall.i) #0
  call void @llvm.lifetime.end.p0i8(i64 0, i8* %213)
  call void @llvm.lifetime.end.p0i8(i64 1920, i8* %214)
  call void @llvm.lifetime.end.p0i8(i64 1152, i8* %215)
  call void @llvm.lifetime.end.p0i8(i64 1920, i8* %216)
  call void @llvm.lifetime.end.p0i8(i64 384, i8* %217)
  call void @llvm.lifetime.end.p0i8(i64 1152, i8* %218)
  call void @llvm.lifetime.end.p0i8(i64 1920, i8* %219)
  call void @llvm.lifetime.end.p0i8(i64 1920, i8* %220)
  call void @llvm.lifetime.end.p0i8(i64 512, i8* %221)
  call void @llvm.lifetime.end.p0i8(i64 1920, i8* %222)
  call void @llvm.lifetime.end.p0i8(i64 512, i8* %223)
  call void @llvm.lifetime.end.p0i8(i64 24, i8* %224)
  call void @llvm.lifetime.end.p0i8(i64 24, i8* %225)
  call void @llvm.lifetime.end.p0i8(i64 64, i8* %226)
  call void @llvm.lifetime.end.p0i8(i64 3, i8* %227)
  call void @llvm.stackrestore(i8* %savedstack)
  ret i32 0
}

; Function Attrs: inaccessiblememonly nofree nounwind willreturn
declare noalias noundef i8* @malloc(i64) local_unnamed_addr #1

; Function Attrs: nounwind
declare void @nnc_prepacked_conv2d_clamp_run(i64, i8**, i64*, i64*, i8*, i64, i64*) local_unnamed_addr #0

; Function Attrs: nounwind
declare void @nnc_aten_adaptive_avg_pool2d(i64, i8**, i64*, i64*, i8*, i64, i64*) local_unnamed_addr #0

; Function Attrs: nounwind
declare void @nnc_prepacked_linear_clamp_run(i64, i8**, i64*, i64*, i8*, i64, i64*) local_unnamed_addr #0

; Function Attrs: inaccessiblemem_or_argmemonly nounwind willreturn
declare void @free(i8* nocapture noundef) local_unnamed_addr #2

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #3

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.experimental.noalias.scope.decl(metadata) #4

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #3

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #3

; Function Attrs: nofree nosync nounwind willreturn
declare i8* @llvm.stacksave() #5

; Function Attrs: nofree nosync nounwind willreturn
declare void @llvm.stackrestore(i8*) #5

attributes #0 = { nounwind }
attributes #1 = { inaccessiblememonly nofree nounwind willreturn }
attributes #2 = { inaccessiblemem_or_argmemonly nounwind willreturn }
attributes #3 = { argmemonly nofree nosync nounwind willreturn }
attributes #4 = { inaccessiblememonly nofree nosync nounwind willreturn }
attributes #5 = { nofree nosync nounwind willreturn }

!0 = !{!1}
!1 = distinct !{!1, !2, !"pytorch: argument 0"}
!2 = distinct !{!2, !"pytorch"}
!3 = !{!4}
!4 = distinct !{!4, !2, !"pytorch: argument 1"}
!5 = !{!6}
!6 = distinct !{!6, !2, !"pytorch: argument 2"}
!7 = !{!8}
!8 = distinct !{!8, !2, !"pytorch: argument 3"}
!9 = !{!10}
!10 = distinct !{!10, !2, !"pytorch: argument 4"}
!11 = !{!12}
!12 = distinct !{!12, !2, !"pytorch: argument 5"}
!13 = !{!14}
!14 = distinct !{!14, !2, !"pytorch: argument 6"}
!15 = !{!16}
!16 = distinct !{!16, !2, !"pytorch: argument 7"}
!17 = !{!18}
!18 = distinct !{!18, !2, !"pytorch: argument 8"}
!19 = !{!20}
!20 = distinct !{!20, !2, !"pytorch: argument 9"}
!21 = !{!22}
!22 = distinct !{!22, !2, !"pytorch: argument 10"}
!23 = !{!24}
!24 = distinct !{!24, !2, !"pytorch: argument 11"}
!25 = !{!26}
!26 = distinct !{!26, !2, !"pytorch: argument 12"}
!27 = !{!28}
!28 = distinct !{!28, !2, !"pytorch: argument 13"}
!29 = !{!30}
!30 = distinct !{!30, !2, !"pytorch: argument 14"}
!31 = !{!32}
!32 = distinct !{!32, !2, !"pytorch: argument 15"}
!33 = !{!34}
!34 = distinct !{!34, !2, !"pytorch: argument 16"}
!35 = !{!36}
!36 = distinct !{!36, !2, !"pytorch: argument 17"}
!37 = !{!38}
!38 = distinct !{!38, !2, !"pytorch: argument 18"}
!39 = !{!40}
!40 = distinct !{!40, !2, !"pytorch: argument 19"}
!41 = !{!42}
!42 = distinct !{!42, !2, !"pytorch: argument 20"}
!43 = !{!44}
!44 = distinct !{!44, !2, !"pytorch: argument 21"}
!45 = !{!46}
!46 = distinct !{!46, !2, !"pytorch: argument 22"}
!47 = !{!48}
!48 = distinct !{!48, !2, !"pytorch: argument 23"}
!49 = !{!50}
!50 = distinct !{!50, !2, !"pytorch: argument 24"}
!51 = !{!52}
!52 = distinct !{!52, !2, !"pytorch: argument 25"}
!53 = !{!54}
!54 = distinct !{!54, !2, !"pytorch: argument 26"}
!55 = !{!56}
!56 = distinct !{!56, !2, !"pytorch: argument 27"}
!57 = !{!58}
!58 = distinct !{!58, !2, !"pytorch: argument 28"}
!59 = !{!60}
!60 = distinct !{!60, !2, !"pytorch: argument 29"}
!61 = !{!62}
!62 = distinct !{!62, !2, !"pytorch: argument 30"}
!63 = !{!64}
!64 = distinct !{!64, !2, !"pytorch: argument 31"}
!65 = !{!66}
!66 = distinct !{!66, !2, !"pytorch: argument 32"}
!67 = !{!68}
!68 = distinct !{!68, !2, !"pytorch: argument 33"}
!69 = !{!70}
!70 = distinct !{!70, !2, !"pytorch: argument 34"}
!71 = !{!72}
!72 = distinct !{!72, !2, !"pytorch: argument 35"}
!73 = !{!74}
!74 = distinct !{!74, !2, !"pytorch: argument 36"}
!75 = !{!76}
!76 = distinct !{!76, !2, !"pytorch: argument 37"}
!77 = !{!78}
!78 = distinct !{!78, !2, !"pytorch: argument 38"}
!79 = !{!80}
!80 = distinct !{!80, !2, !"pytorch: argument 39"}
!81 = !{!82}
!82 = distinct !{!82, !2, !"pytorch: argument 40"}
!83 = !{!84}
!84 = distinct !{!84, !2, !"pytorch: argument 41"}
!85 = !{!86}
!86 = distinct !{!86, !2, !"pytorch: argument 42"}
!87 = !{!88}
!88 = distinct !{!88, !2, !"pytorch: argument 43"}
!89 = !{!90}
!90 = distinct !{!90, !2, !"pytorch: argument 44"}
!91 = !{!92}
!92 = distinct !{!92, !2, !"pytorch: argument 45"}
!93 = !{!94}
!94 = distinct !{!94, !2, !"pytorch: argument 46"}
!95 = !{!96}
!96 = distinct !{!96, !2, !"pytorch: argument 47"}
!97 = !{!98}
!98 = distinct !{!98, !2, !"pytorch: argument 48"}
!99 = !{!100}
!100 = distinct !{!100, !2, !"pytorch: argument 49"}
!101 = !{!102}
!102 = distinct !{!102, !2, !"pytorch: argument 50"}
!103 = !{!104}
!104 = distinct !{!104, !2, !"pytorch: argument 51"}
!105 = !{!106}
!106 = distinct !{!106, !2, !"pytorch: argument 52"}
!107 = !{!108}
!108 = distinct !{!108, !2, !"pytorch: argument 53"}
!109 = !{!110}
!110 = distinct !{!110, !2, !"pytorch: argument 54"}
!111 = !{!112}
!112 = distinct !{!112, !2, !"pytorch: argument 55"}
!113 = !{!114}
!114 = distinct !{!114, !2, !"pytorch: argument 56"}
!115 = !{!116}
!116 = distinct !{!116, !2, !"pytorch: argument 57"}
!117 = !{!118}
!118 = distinct !{!118, !2, !"pytorch: argument 58"}
!119 = !{!120}
!120 = distinct !{!120, !2, !"pytorch: argument 59"}
!121 = !{!122}
!122 = distinct !{!122, !2, !"pytorch: argument 60"}
!123 = !{!124}
!124 = distinct !{!124, !2, !"pytorch: argument 61"}
!125 = !{!126}
!126 = distinct !{!126, !2, !"pytorch: argument 62"}
!127 = !{!128}
!128 = distinct !{!128, !2, !"pytorch: argument 63"}
!129 = !{!130}
!130 = distinct !{!130, !2, !"pytorch: argument 64"}
!131 = !{!132}
!132 = distinct !{!132, !2, !"pytorch: argument 65"}
!133 = !{!1, !4, !6, !8, !10, !12, !14, !16, !18, !20, !22, !24, !26, !28, !30, !32, !34, !36, !38, !40, !42, !44, !46, !48, !50, !52, !54, !56, !58, !60, !62, !64, !66, !68, !70, !72, !74, !76, !78, !80, !82, !84, !86, !88, !90, !92, !94, !96, !98, !100, !102, !104, !106, !108, !110, !112, !114, !116, !118, !120, !122, !124, !126, !128, !130, !132}
!134 = !{!4, !8, !10, !12, !14, !16, !18, !20, !22, !24, !26, !28, !30, !32, !34, !36, !38, !40, !42, !44, !46, !48, !50, !52, !54, !56, !58, !60, !62, !64, !66, !68, !70, !72, !74, !76, !78, !80, !82, !84, !86, !88, !90, !92, !94, !96, !98, !100, !102, !104, !106, !108, !110, !112, !114, !116, !118, !120, !122, !124, !126, !128, !130, !132}
!135 = !{!4, !110, !112, !114, !116, !118, !120, !122, !124, !126, !128, !130, !132}
!136 = !{!4, !112, !114, !116, !118, !120, !122, !124, !126, !128, !130, !132}
!137 = !{!4, !114, !116, !118, !120, !122, !124, !126, !128, !130, !132}
!138 = !{!4, !116, !118, !120, !122, !124, !126, !128, !130, !132}
!139 = !{!4, !118, !120, !122, !124, !126, !128, !130, !132}
!140 = !{!4, !120, !122, !124, !126, !128, !130, !132}
!141 = !{!4, !122, !124, !126, !128, !130, !132}
!142 = !{!4, !124, !126, !128, !130, !132}
!143 = !{!4, !126, !128, !130, !132}
!144 = !{!4, !128, !130, !132}
!145 = !{!4, !130, !132}
!146 = !{!4, !132}
!147 = distinct !{!147, !148}
!148 = !{!"llvm.loop.isvectorized", i32 1}
