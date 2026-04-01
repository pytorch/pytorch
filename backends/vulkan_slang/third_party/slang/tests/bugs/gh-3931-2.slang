//IGNORE:TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -compute -shaderobj -xslang -matrix-layout-row-major -xslang -O0
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -compute -shaderobj -xslang -matrix-layout-row-major -xslang -O0 -xslang -DWORKAROUND

// removes from code all uses of pointers, this fixes the code from removal of `PhysicalStorageBuffer64` -- note that the pointer branch is unused regardless of if toggled
//#define WORKAROUND 

// ensures the matrix byte offset is 0, this 'fixes' the issue since the issue is that member memory offsets were not being respected (and offset 0 was loaded for each struct)
//#define WORKAROUND2 
struct FooBar {
#ifdef WORKAROUND2
    float4x4 c;
    uint64_t a;
    uint32_t b;
#else
    uint64_t a;
    uint32_t b;
    float4x4 c;
#endif

    int load() {
#ifdef WORKAROUND
        return (int)a;
#else
        return *((int *)a);
#endif
    }
};

//TEST_INPUT: ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

//TEST_INPUT:ubuffer(data=[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400], stride=4):name=sb
uniform StructuredBuffer<FooBar, ScalarDataLayout> sb;

int test(int val)
{
    uint32_t res = 0;
    for (int i = 0; i < 4; ++i) {
        res |= bit_cast<uint32_t>(sb[val].c[0][i]) << i * 8;
    }

    if (val == 100000) { // will never use a pointer
        res = sb[val].load();
    }
    return bit_cast<int>(res);
}

[numthreads(4, 1, 1)]
[shader("compute")]
void computeMain(
    int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;

    int inVal = tid;
    int outVal = test(inVal);

    // CHECK: 6050403
    // CHECK: 1A191817
    // CHECK: 2E2D2C2B
    // CHECK: 4241403F
    outputBuffer[tid] = outVal;
}