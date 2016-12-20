/*
Language: AutoIt
Author: Manh Tuan <junookyo@gmail.com>
Description: AutoIt language definition
Category: scripting
*/

function(hljs) {
    var KEYWORDS = 'ByRef Case Const ContinueCase ContinueLoop ' +
        'Default Dim Do Else ElseIf EndFunc EndIf EndSelect ' +
        'EndSwitch EndWith Enum Exit ExitLoop For Func ' +
        'Global If In Local Next ReDim Return Select Static ' +
        'Step Switch Then To Until Volatile WEnd While With',

        LITERAL = 'True False And Null Not Or',

        BUILT_IN = 'Abs ACos AdlibRegister AdlibUnRegister Asc AscW ASin ' +
        'Assign ATan AutoItSetOption AutoItWinGetTitle ' +
        'AutoItWinSetTitle Beep Binary BinaryLen BinaryMid ' +
        'BinaryToString BitAND BitNOT BitOR BitRotate BitShift ' +
        'BitXOR BlockInput Break Call CDTray Ceiling Chr ' +
        'ChrW ClipGet ClipPut ConsoleRead ConsoleWrite ' +
        'ConsoleWriteError ControlClick ControlCommand ' +
        'ControlDisable ControlEnable ControlFocus ControlGetFocus ' +
        'ControlGetHandle ControlGetPos ControlGetText ControlHide ' +
        'ControlListView ControlMove ControlSend ControlSetText ' +
        'ControlShow ControlTreeView Cos Dec DirCopy DirCreate ' +
        'DirGetSize DirMove DirRemove DllCall DllCallAddress ' +
        'DllCallbackFree DllCallbackGetPtr DllCallbackRegister ' +
        'DllClose DllOpen DllStructCreate DllStructGetData ' +
        'DllStructGetPtr DllStructGetSize DllStructSetData ' +
        'DriveGetDrive DriveGetFileSystem DriveGetLabel ' +
        'DriveGetSerial DriveGetType DriveMapAdd DriveMapDel ' +
        'DriveMapGet DriveSetLabel DriveSpaceFree DriveSpaceTotal ' +
        'DriveStatus EnvGet EnvSet EnvUpdate Eval Execute Exp ' +
        'FileChangeDir FileClose FileCopy FileCreateNTFSLink ' +
        'FileCreateShortcut FileDelete FileExists FileFindFirstFile ' +
        'FileFindNextFile FileFlush FileGetAttrib FileGetEncoding ' +
        'FileGetLongName FileGetPos FileGetShortcut FileGetShortName ' +
        'FileGetSize FileGetTime FileGetVersion FileInstall ' +
        'FileMove FileOpen FileOpenDialog FileRead FileReadLine ' +
        'FileReadToArray FileRecycle FileRecycleEmpty FileSaveDialog ' +
        'FileSelectFolder FileSetAttrib FileSetEnd FileSetPos ' +
        'FileSetTime FileWrite FileWriteLine Floor FtpSetProxy ' +
        'FuncName GUICreate GUICtrlCreateAvi GUICtrlCreateButton ' +
        'GUICtrlCreateCheckbox GUICtrlCreateCombo ' +
        'GUICtrlCreateContextMenu GUICtrlCreateDate GUICtrlCreateDummy ' +
        'GUICtrlCreateEdit GUICtrlCreateGraphic GUICtrlCreateGroup ' +
        'GUICtrlCreateIcon GUICtrlCreateInput GUICtrlCreateLabel ' +
        'GUICtrlCreateList GUICtrlCreateListView ' +
        'GUICtrlCreateListViewItem GUICtrlCreateMenu ' +
        'GUICtrlCreateMenuItem GUICtrlCreateMonthCal GUICtrlCreateObj ' +
        'GUICtrlCreatePic GUICtrlCreateProgress GUICtrlCreateRadio ' +
        'GUICtrlCreateSlider GUICtrlCreateTab GUICtrlCreateTabItem ' +
        'GUICtrlCreateTreeView GUICtrlCreateTreeViewItem ' +
        'GUICtrlCreateUpdown GUICtrlDelete GUICtrlGetHandle ' +
        'GUICtrlGetState GUICtrlRead GUICtrlRecvMsg ' +
        'GUICtrlRegisterListViewSort GUICtrlSendMsg GUICtrlSendToDummy ' +
        'GUICtrlSetBkColor GUICtrlSetColor GUICtrlSetCursor ' +
        'GUICtrlSetData GUICtrlSetDefBkColor GUICtrlSetDefColor ' +
        'GUICtrlSetFont GUICtrlSetGraphic GUICtrlSetImage ' +
        'GUICtrlSetLimit GUICtrlSetOnEvent GUICtrlSetPos ' +
        'GUICtrlSetResizing GUICtrlSetState GUICtrlSetStyle ' +
        'GUICtrlSetTip GUIDelete GUIGetCursorInfo GUIGetMsg ' +
        'GUIGetStyle GUIRegisterMsg GUISetAccelerators GUISetBkColor ' +
        'GUISetCoord GUISetCursor GUISetFont GUISetHelp GUISetIcon ' +
        'GUISetOnEvent GUISetState GUISetStyle GUIStartGroup ' +
        'GUISwitch Hex HotKeySet HttpSetProxy HttpSetUserAgent ' +
        'HWnd InetClose InetGet InetGetInfo InetGetSize InetRead ' +
        'IniDelete IniRead IniReadSection IniReadSectionNames ' +
        'IniRenameSection IniWrite IniWriteSection InputBox Int ' +
        'IsAdmin IsArray IsBinary IsBool IsDeclared IsDllStruct ' +
        'IsFloat IsFunc IsHWnd IsInt IsKeyword IsNumber IsObj ' +
        'IsPtr IsString Log MemGetStats Mod MouseClick ' +
        'MouseClickDrag MouseDown MouseGetCursor MouseGetPos ' +
        'MouseMove MouseUp MouseWheel MsgBox Number ObjCreate ' +
        'ObjCreateInterface ObjEvent ObjGet ObjName ' +
        'OnAutoItExitRegister OnAutoItExitUnRegister Opt Ping ' +
        'PixelChecksum PixelGetColor PixelSearch ProcessClose ' +
        'ProcessExists ProcessGetStats ProcessList ' +
        'ProcessSetPriority ProcessWait ProcessWaitClose ProgressOff ' +
        'ProgressOn ProgressSet Ptr Random RegDelete RegEnumKey ' +
        'RegEnumVal RegRead RegWrite Round Run RunAs RunAsWait ' +
        'RunWait Send SendKeepActive SetError SetExtended ' +
        'ShellExecute ShellExecuteWait Shutdown Sin Sleep ' +
        'SoundPlay SoundSetWaveVolume SplashImageOn SplashOff ' +
        'SplashTextOn Sqrt SRandom StatusbarGetText StderrRead ' +
        'StdinWrite StdioClose StdoutRead String StringAddCR ' +
        'StringCompare StringFormat StringFromASCIIArray StringInStr ' +
        'StringIsAlNum StringIsAlpha StringIsASCII StringIsDigit ' +
        'StringIsFloat StringIsInt StringIsLower StringIsSpace ' +
        'StringIsUpper StringIsXDigit StringLeft StringLen ' +
        'StringLower StringMid StringRegExp StringRegExpReplace ' +
        'StringReplace StringReverse StringRight StringSplit ' +
        'StringStripCR StringStripWS StringToASCIIArray ' +
        'StringToBinary StringTrimLeft StringTrimRight StringUpper ' +
        'Tan TCPAccept TCPCloseSocket TCPConnect TCPListen ' +
        'TCPNameToIP TCPRecv TCPSend TCPShutdown TCPStartup ' +
        'TimerDiff TimerInit ToolTip TrayCreateItem TrayCreateMenu ' +
        'TrayGetMsg TrayItemDelete TrayItemGetHandle ' +
        'TrayItemGetState TrayItemGetText TrayItemSetOnEvent ' +
        'TrayItemSetState TrayItemSetText TraySetClick TraySetIcon ' +
        'TraySetOnEvent TraySetPauseIcon TraySetState TraySetToolTip ' +
        'TrayTip UBound UDPBind UDPCloseSocket UDPOpen UDPRecv ' +
        'UDPSend UDPShutdown UDPStartup VarGetType WinActivate ' +
        'WinActive WinClose WinExists WinFlash WinGetCaretPos ' +
        'WinGetClassList WinGetClientSize WinGetHandle WinGetPos ' +
        'WinGetProcess WinGetState WinGetText WinGetTitle WinKill ' +
        'WinList WinMenuSelectItem WinMinimizeAll WinMinimizeAllUndo ' +
        'WinMove WinSetOnTop WinSetState WinSetTitle WinSetTrans ' +
        'WinWait WinWaitActive WinWaitClose WinWaitNotActive ' +
        'Array1DToHistogram ArrayAdd ArrayBinarySearch ' +
        'ArrayColDelete ArrayColInsert ArrayCombinations ' +
        'ArrayConcatenate ArrayDelete ArrayDisplay ArrayExtract ' +
        'ArrayFindAll ArrayInsert ArrayMax ArrayMaxIndex ArrayMin ' +
        'ArrayMinIndex ArrayPermute ArrayPop ArrayPush ' +
        'ArrayReverse ArraySearch ArrayShuffle ArraySort ArraySwap ' +
        'ArrayToClip ArrayToString ArrayTranspose ArrayTrim ' +
        'ArrayUnique Assert ChooseColor ChooseFont ' +
        'ClipBoard_ChangeChain ClipBoard_Close ClipBoard_CountFormats ' +
        'ClipBoard_Empty ClipBoard_EnumFormats ClipBoard_FormatStr ' +
        'ClipBoard_GetData ClipBoard_GetDataEx ClipBoard_GetFormatName ' +
        'ClipBoard_GetOpenWindow ClipBoard_GetOwner ' +
        'ClipBoard_GetPriorityFormat ClipBoard_GetSequenceNumber ' +
        'ClipBoard_GetViewer ClipBoard_IsFormatAvailable ' +
        'ClipBoard_Open ClipBoard_RegisterFormat ClipBoard_SetData ' +
        'ClipBoard_SetDataEx ClipBoard_SetViewer ClipPutFile ' +
        'ColorConvertHSLtoRGB ColorConvertRGBtoHSL ColorGetBlue ' +
        'ColorGetCOLORREF ColorGetGreen ColorGetRed ColorGetRGB ' +
        'ColorSetCOLORREF ColorSetRGB Crypt_DecryptData ' +
        'Crypt_DecryptFile Crypt_DeriveKey Crypt_DestroyKey ' +
        'Crypt_EncryptData Crypt_EncryptFile Crypt_GenRandom ' +
        'Crypt_HashData Crypt_HashFile Crypt_Shutdown Crypt_Startup ' +
        'DateAdd DateDayOfWeek DateDaysInMonth DateDiff ' +
        'DateIsLeapYear DateIsValid DateTimeFormat DateTimeSplit ' +
        'DateToDayOfWeek DateToDayOfWeekISO DateToDayValue ' +
        'DateToMonth Date_Time_CompareFileTime ' +
        'Date_Time_DOSDateTimeToArray Date_Time_DOSDateTimeToFileTime ' +
        'Date_Time_DOSDateTimeToStr Date_Time_DOSDateToArray ' +
        'Date_Time_DOSDateToStr Date_Time_DOSTimeToArray ' +
        'Date_Time_DOSTimeToStr Date_Time_EncodeFileTime ' +
        'Date_Time_EncodeSystemTime Date_Time_FileTimeToArray ' +
        'Date_Time_FileTimeToDOSDateTime ' +
        'Date_Time_FileTimeToLocalFileTime Date_Time_FileTimeToStr ' +
        'Date_Time_FileTimeToSystemTime Date_Time_GetFileTime ' +
        'Date_Time_GetLocalTime Date_Time_GetSystemTime ' +
        'Date_Time_GetSystemTimeAdjustment ' +
        'Date_Time_GetSystemTimeAsFileTime Date_Time_GetSystemTimes ' +
        'Date_Time_GetTickCount Date_Time_GetTimeZoneInformation ' +
        'Date_Time_LocalFileTimeToFileTime Date_Time_SetFileTime ' +
        'Date_Time_SetLocalTime Date_Time_SetSystemTime ' +
        'Date_Time_SetSystemTimeAdjustment ' +
        'Date_Time_SetTimeZoneInformation Date_Time_SystemTimeToArray ' +
        'Date_Time_SystemTimeToDateStr Date_Time_SystemTimeToDateTimeStr ' +
        'Date_Time_SystemTimeToFileTime Date_Time_SystemTimeToTimeStr ' +
        'Date_Time_SystemTimeToTzSpecificLocalTime ' +
        'Date_Time_TzSpecificLocalTimeToSystemTime DayValueToDate ' +
        'DebugBugReportEnv DebugCOMError DebugOut DebugReport ' +
        'DebugReportEx DebugReportVar DebugSetup Degree ' +
        'EventLog__Backup EventLog__Clear EventLog__Close ' +
        'EventLog__Count EventLog__DeregisterSource EventLog__Full ' +
        'EventLog__Notify EventLog__Oldest EventLog__Open ' +
        'EventLog__OpenBackup EventLog__Read EventLog__RegisterSource ' +
        'EventLog__Report Excel_BookAttach Excel_BookClose ' +
        'Excel_BookList Excel_BookNew Excel_BookOpen ' +
        'Excel_BookOpenText Excel_BookSave Excel_BookSaveAs ' +
        'Excel_Close Excel_ColumnToLetter Excel_ColumnToNumber ' +
        'Excel_ConvertFormula Excel_Export Excel_FilterGet ' +
        'Excel_FilterSet Excel_Open Excel_PictureAdd Excel_Print ' +
        'Excel_RangeCopyPaste Excel_RangeDelete Excel_RangeFind ' +
        'Excel_RangeInsert Excel_RangeLinkAddRemove Excel_RangeRead ' +
        'Excel_RangeReplace Excel_RangeSort Excel_RangeValidate ' +
        'Excel_RangeWrite Excel_SheetAdd Excel_SheetCopyMove ' +
        'Excel_SheetDelete Excel_SheetList FileCountLines FileCreate ' +
        'FileListToArray FileListToArrayRec FilePrint ' +
        'FileReadToArray FileWriteFromArray FileWriteLog ' +
        'FileWriteToLine FTP_Close FTP_Command FTP_Connect ' +
        'FTP_DecodeInternetStatus FTP_DirCreate FTP_DirDelete ' +
        'FTP_DirGetCurrent FTP_DirPutContents FTP_DirSetCurrent ' +
        'FTP_FileClose FTP_FileDelete FTP_FileGet FTP_FileGetSize ' +
        'FTP_FileOpen FTP_FilePut FTP_FileRead FTP_FileRename ' +
        'FTP_FileTimeLoHiToStr FTP_FindFileClose FTP_FindFileFirst ' +
        'FTP_FindFileNext FTP_GetLastResponseInfo FTP_ListToArray ' +
        'FTP_ListToArray2D FTP_ListToArrayEx FTP_Open ' +
        'FTP_ProgressDownload FTP_ProgressUpload FTP_SetStatusCallback ' +
        'GDIPlus_ArrowCapCreate GDIPlus_ArrowCapDispose ' +
        'GDIPlus_ArrowCapGetFillState GDIPlus_ArrowCapGetHeight ' +
        'GDIPlus_ArrowCapGetMiddleInset GDIPlus_ArrowCapGetWidth ' +
        'GDIPlus_ArrowCapSetFillState GDIPlus_ArrowCapSetHeight ' +
        'GDIPlus_ArrowCapSetMiddleInset GDIPlus_ArrowCapSetWidth ' +
        'GDIPlus_BitmapApplyEffect GDIPlus_BitmapApplyEffectEx ' +
        'GDIPlus_BitmapCloneArea GDIPlus_BitmapConvertFormat ' +
        'GDIPlus_BitmapCreateApplyEffect ' +
        'GDIPlus_BitmapCreateApplyEffectEx ' +
        'GDIPlus_BitmapCreateDIBFromBitmap GDIPlus_BitmapCreateFromFile ' +
        'GDIPlus_BitmapCreateFromGraphics ' +
        'GDIPlus_BitmapCreateFromHBITMAP GDIPlus_BitmapCreateFromHICON ' +
        'GDIPlus_BitmapCreateFromHICON32 GDIPlus_BitmapCreateFromMemory ' +
        'GDIPlus_BitmapCreateFromResource GDIPlus_BitmapCreateFromScan0 ' +
        'GDIPlus_BitmapCreateFromStream ' +
        'GDIPlus_BitmapCreateHBITMAPFromBitmap GDIPlus_BitmapDispose ' +
        'GDIPlus_BitmapGetHistogram GDIPlus_BitmapGetHistogramEx ' +
        'GDIPlus_BitmapGetHistogramSize GDIPlus_BitmapGetPixel ' +
        'GDIPlus_BitmapLockBits GDIPlus_BitmapSetPixel ' +
        'GDIPlus_BitmapUnlockBits GDIPlus_BrushClone ' +
        'GDIPlus_BrushCreateSolid GDIPlus_BrushDispose ' +
        'GDIPlus_BrushGetSolidColor GDIPlus_BrushGetType ' +
        'GDIPlus_BrushSetSolidColor GDIPlus_ColorMatrixCreate ' +
        'GDIPlus_ColorMatrixCreateGrayScale ' +
        'GDIPlus_ColorMatrixCreateNegative ' +
        'GDIPlus_ColorMatrixCreateSaturation ' +
        'GDIPlus_ColorMatrixCreateScale ' +
        'GDIPlus_ColorMatrixCreateTranslate GDIPlus_CustomLineCapClone ' +
        'GDIPlus_CustomLineCapCreate GDIPlus_CustomLineCapDispose ' +
        'GDIPlus_CustomLineCapGetStrokeCaps ' +
        'GDIPlus_CustomLineCapSetStrokeCaps GDIPlus_Decoders ' +
        'GDIPlus_DecodersGetCount GDIPlus_DecodersGetSize ' +
        'GDIPlus_DrawImageFX GDIPlus_DrawImageFXEx ' +
        'GDIPlus_DrawImagePoints GDIPlus_EffectCreate ' +
        'GDIPlus_EffectCreateBlur GDIPlus_EffectCreateBrightnessContrast ' +
        'GDIPlus_EffectCreateColorBalance GDIPlus_EffectCreateColorCurve ' +
        'GDIPlus_EffectCreateColorLUT GDIPlus_EffectCreateColorMatrix ' +
        'GDIPlus_EffectCreateHueSaturationLightness ' +
        'GDIPlus_EffectCreateLevels GDIPlus_EffectCreateRedEyeCorrection ' +
        'GDIPlus_EffectCreateSharpen GDIPlus_EffectCreateTint ' +
        'GDIPlus_EffectDispose GDIPlus_EffectGetParameters ' +
        'GDIPlus_EffectSetParameters GDIPlus_Encoders ' +
        'GDIPlus_EncodersGetCLSID GDIPlus_EncodersGetCount ' +
        'GDIPlus_EncodersGetParamList GDIPlus_EncodersGetParamListSize ' +
        'GDIPlus_EncodersGetSize GDIPlus_FontCreate ' +
        'GDIPlus_FontDispose GDIPlus_FontFamilyCreate ' +
        'GDIPlus_FontFamilyCreateFromCollection ' +
        'GDIPlus_FontFamilyDispose GDIPlus_FontFamilyGetCellAscent ' +
        'GDIPlus_FontFamilyGetCellDescent GDIPlus_FontFamilyGetEmHeight ' +
        'GDIPlus_FontFamilyGetLineSpacing GDIPlus_FontGetHeight ' +
        'GDIPlus_FontPrivateAddFont GDIPlus_FontPrivateAddMemoryFont ' +
        'GDIPlus_FontPrivateCollectionDispose ' +
        'GDIPlus_FontPrivateCreateCollection GDIPlus_GraphicsClear ' +
        'GDIPlus_GraphicsCreateFromHDC GDIPlus_GraphicsCreateFromHWND ' +
        'GDIPlus_GraphicsDispose GDIPlus_GraphicsDrawArc ' +
        'GDIPlus_GraphicsDrawBezier GDIPlus_GraphicsDrawClosedCurve ' +
        'GDIPlus_GraphicsDrawClosedCurve2 GDIPlus_GraphicsDrawCurve ' +
        'GDIPlus_GraphicsDrawCurve2 GDIPlus_GraphicsDrawEllipse ' +
        'GDIPlus_GraphicsDrawImage GDIPlus_GraphicsDrawImagePointsRect ' +
        'GDIPlus_GraphicsDrawImageRect GDIPlus_GraphicsDrawImageRectRect ' +
        'GDIPlus_GraphicsDrawLine GDIPlus_GraphicsDrawPath ' +
        'GDIPlus_GraphicsDrawPie GDIPlus_GraphicsDrawPolygon ' +
        'GDIPlus_GraphicsDrawRect GDIPlus_GraphicsDrawString ' +
        'GDIPlus_GraphicsDrawStringEx GDIPlus_GraphicsFillClosedCurve ' +
        'GDIPlus_GraphicsFillClosedCurve2 GDIPlus_GraphicsFillEllipse ' +
        'GDIPlus_GraphicsFillPath GDIPlus_GraphicsFillPie ' +
        'GDIPlus_GraphicsFillPolygon GDIPlus_GraphicsFillRect ' +
        'GDIPlus_GraphicsFillRegion GDIPlus_GraphicsGetCompositingMode ' +
        'GDIPlus_GraphicsGetCompositingQuality GDIPlus_GraphicsGetDC ' +
        'GDIPlus_GraphicsGetInterpolationMode ' +
        'GDIPlus_GraphicsGetSmoothingMode GDIPlus_GraphicsGetTransform ' +
        'GDIPlus_GraphicsMeasureCharacterRanges ' +
        'GDIPlus_GraphicsMeasureString GDIPlus_GraphicsReleaseDC ' +
        'GDIPlus_GraphicsResetClip GDIPlus_GraphicsResetTransform ' +
        'GDIPlus_GraphicsRestore GDIPlus_GraphicsRotateTransform ' +
        'GDIPlus_GraphicsSave GDIPlus_GraphicsScaleTransform ' +
        'GDIPlus_GraphicsSetClipPath GDIPlus_GraphicsSetClipRect ' +
        'GDIPlus_GraphicsSetClipRegion ' +
        'GDIPlus_GraphicsSetCompositingMode ' +
        'GDIPlus_GraphicsSetCompositingQuality ' +
        'GDIPlus_GraphicsSetInterpolationMode ' +
        'GDIPlus_GraphicsSetPixelOffsetMode ' +
        'GDIPlus_GraphicsSetSmoothingMode ' +
        'GDIPlus_GraphicsSetTextRenderingHint ' +
        'GDIPlus_GraphicsSetTransform GDIPlus_GraphicsTransformPoints ' +
        'GDIPlus_GraphicsTranslateTransform GDIPlus_HatchBrushCreate ' +
        'GDIPlus_HICONCreateFromBitmap GDIPlus_ImageAttributesCreate ' +
        'GDIPlus_ImageAttributesDispose ' +
        'GDIPlus_ImageAttributesSetColorKeys ' +
        'GDIPlus_ImageAttributesSetColorMatrix GDIPlus_ImageDispose ' +
        'GDIPlus_ImageGetDimension GDIPlus_ImageGetFlags ' +
        'GDIPlus_ImageGetGraphicsContext GDIPlus_ImageGetHeight ' +
        'GDIPlus_ImageGetHorizontalResolution ' +
        'GDIPlus_ImageGetPixelFormat GDIPlus_ImageGetRawFormat ' +
        'GDIPlus_ImageGetThumbnail GDIPlus_ImageGetType ' +
        'GDIPlus_ImageGetVerticalResolution GDIPlus_ImageGetWidth ' +
        'GDIPlus_ImageLoadFromFile GDIPlus_ImageLoadFromStream ' +
        'GDIPlus_ImageResize GDIPlus_ImageRotateFlip ' +
        'GDIPlus_ImageSaveToFile GDIPlus_ImageSaveToFileEx ' +
        'GDIPlus_ImageSaveToStream GDIPlus_ImageScale ' +
        'GDIPlus_LineBrushCreate GDIPlus_LineBrushCreateFromRect ' +
        'GDIPlus_LineBrushCreateFromRectWithAngle ' +
        'GDIPlus_LineBrushGetColors GDIPlus_LineBrushGetRect ' +
        'GDIPlus_LineBrushMultiplyTransform ' +
        'GDIPlus_LineBrushResetTransform GDIPlus_LineBrushSetBlend ' +
        'GDIPlus_LineBrushSetColors GDIPlus_LineBrushSetGammaCorrection ' +
        'GDIPlus_LineBrushSetLinearBlend GDIPlus_LineBrushSetPresetBlend ' +
        'GDIPlus_LineBrushSetSigmaBlend GDIPlus_LineBrushSetTransform ' +
        'GDIPlus_MatrixClone GDIPlus_MatrixCreate ' +
        'GDIPlus_MatrixDispose GDIPlus_MatrixGetElements ' +
        'GDIPlus_MatrixInvert GDIPlus_MatrixMultiply ' +
        'GDIPlus_MatrixRotate GDIPlus_MatrixScale ' +
        'GDIPlus_MatrixSetElements GDIPlus_MatrixShear ' +
        'GDIPlus_MatrixTransformPoints GDIPlus_MatrixTranslate ' +
        'GDIPlus_PaletteInitialize GDIPlus_ParamAdd GDIPlus_ParamInit ' +
        'GDIPlus_ParamSize GDIPlus_PathAddArc GDIPlus_PathAddBezier ' +
        'GDIPlus_PathAddClosedCurve GDIPlus_PathAddClosedCurve2 ' +
        'GDIPlus_PathAddCurve GDIPlus_PathAddCurve2 ' +
        'GDIPlus_PathAddCurve3 GDIPlus_PathAddEllipse ' +
        'GDIPlus_PathAddLine GDIPlus_PathAddLine2 GDIPlus_PathAddPath ' +
        'GDIPlus_PathAddPie GDIPlus_PathAddPolygon ' +
        'GDIPlus_PathAddRectangle GDIPlus_PathAddString ' +
        'GDIPlus_PathBrushCreate GDIPlus_PathBrushCreateFromPath ' +
        'GDIPlus_PathBrushGetCenterPoint GDIPlus_PathBrushGetFocusScales ' +
        'GDIPlus_PathBrushGetPointCount GDIPlus_PathBrushGetRect ' +
        'GDIPlus_PathBrushGetWrapMode GDIPlus_PathBrushMultiplyTransform ' +
        'GDIPlus_PathBrushResetTransform GDIPlus_PathBrushSetBlend ' +
        'GDIPlus_PathBrushSetCenterColor GDIPlus_PathBrushSetCenterPoint ' +
        'GDIPlus_PathBrushSetFocusScales ' +
        'GDIPlus_PathBrushSetGammaCorrection ' +
        'GDIPlus_PathBrushSetLinearBlend GDIPlus_PathBrushSetPresetBlend ' +
        'GDIPlus_PathBrushSetSigmaBlend ' +
        'GDIPlus_PathBrushSetSurroundColor ' +
        'GDIPlus_PathBrushSetSurroundColorsWithCount ' +
        'GDIPlus_PathBrushSetTransform GDIPlus_PathBrushSetWrapMode ' +
        'GDIPlus_PathClone GDIPlus_PathCloseFigure GDIPlus_PathCreate ' +
        'GDIPlus_PathCreate2 GDIPlus_PathDispose GDIPlus_PathFlatten ' +
        'GDIPlus_PathGetData GDIPlus_PathGetFillMode ' +
        'GDIPlus_PathGetLastPoint GDIPlus_PathGetPointCount ' +
        'GDIPlus_PathGetPoints GDIPlus_PathGetWorldBounds ' +
        'GDIPlus_PathIsOutlineVisiblePoint GDIPlus_PathIsVisiblePoint ' +
        'GDIPlus_PathIterCreate GDIPlus_PathIterDispose ' +
        'GDIPlus_PathIterGetSubpathCount GDIPlus_PathIterNextMarkerPath ' +
        'GDIPlus_PathIterNextSubpathPath GDIPlus_PathIterRewind ' +
        'GDIPlus_PathReset GDIPlus_PathReverse GDIPlus_PathSetFillMode ' +
        'GDIPlus_PathSetMarker GDIPlus_PathStartFigure ' +
        'GDIPlus_PathTransform GDIPlus_PathWarp GDIPlus_PathWiden ' +
        'GDIPlus_PathWindingModeOutline GDIPlus_PenCreate ' +
        'GDIPlus_PenCreate2 GDIPlus_PenDispose GDIPlus_PenGetAlignment ' +
        'GDIPlus_PenGetColor GDIPlus_PenGetCustomEndCap ' +
        'GDIPlus_PenGetDashCap GDIPlus_PenGetDashStyle ' +
        'GDIPlus_PenGetEndCap GDIPlus_PenGetMiterLimit ' +
        'GDIPlus_PenGetWidth GDIPlus_PenSetAlignment ' +
        'GDIPlus_PenSetColor GDIPlus_PenSetCustomEndCap ' +
        'GDIPlus_PenSetDashCap GDIPlus_PenSetDashStyle ' +
        'GDIPlus_PenSetEndCap GDIPlus_PenSetLineCap ' +
        'GDIPlus_PenSetLineJoin GDIPlus_PenSetMiterLimit ' +
        'GDIPlus_PenSetStartCap GDIPlus_PenSetWidth ' +
        'GDIPlus_RectFCreate GDIPlus_RegionClone ' +
        'GDIPlus_RegionCombinePath GDIPlus_RegionCombineRect ' +
        'GDIPlus_RegionCombineRegion GDIPlus_RegionCreate ' +
        'GDIPlus_RegionCreateFromPath GDIPlus_RegionCreateFromRect ' +
        'GDIPlus_RegionDispose GDIPlus_RegionGetBounds ' +
        'GDIPlus_RegionGetHRgn GDIPlus_RegionTransform ' +
        'GDIPlus_RegionTranslate GDIPlus_Shutdown GDIPlus_Startup ' +
        'GDIPlus_StringFormatCreate GDIPlus_StringFormatDispose ' +
        'GDIPlus_StringFormatGetMeasurableCharacterRangeCount ' +
        'GDIPlus_StringFormatSetAlign GDIPlus_StringFormatSetLineAlign ' +
        'GDIPlus_StringFormatSetMeasurableCharacterRanges ' +
        'GDIPlus_TextureCreate GDIPlus_TextureCreate2 ' +
        'GDIPlus_TextureCreateIA GetIP GUICtrlAVI_Close ' +
        'GUICtrlAVI_Create GUICtrlAVI_Destroy GUICtrlAVI_IsPlaying ' +
        'GUICtrlAVI_Open GUICtrlAVI_OpenEx GUICtrlAVI_Play ' +
        'GUICtrlAVI_Seek GUICtrlAVI_Show GUICtrlAVI_Stop ' +
        'GUICtrlButton_Click GUICtrlButton_Create ' +
        'GUICtrlButton_Destroy GUICtrlButton_Enable ' +
        'GUICtrlButton_GetCheck GUICtrlButton_GetFocus ' +
        'GUICtrlButton_GetIdealSize GUICtrlButton_GetImage ' +
        'GUICtrlButton_GetImageList GUICtrlButton_GetNote ' +
        'GUICtrlButton_GetNoteLength GUICtrlButton_GetSplitInfo ' +
        'GUICtrlButton_GetState GUICtrlButton_GetText ' +
        'GUICtrlButton_GetTextMargin GUICtrlButton_SetCheck ' +
        'GUICtrlButton_SetDontClick GUICtrlButton_SetFocus ' +
        'GUICtrlButton_SetImage GUICtrlButton_SetImageList ' +
        'GUICtrlButton_SetNote GUICtrlButton_SetShield ' +
        'GUICtrlButton_SetSize GUICtrlButton_SetSplitInfo ' +
        'GUICtrlButton_SetState GUICtrlButton_SetStyle ' +
        'GUICtrlButton_SetText GUICtrlButton_SetTextMargin ' +
        'GUICtrlButton_Show GUICtrlComboBoxEx_AddDir ' +
        'GUICtrlComboBoxEx_AddString GUICtrlComboBoxEx_BeginUpdate ' +
        'GUICtrlComboBoxEx_Create GUICtrlComboBoxEx_CreateSolidBitMap ' +
        'GUICtrlComboBoxEx_DeleteString GUICtrlComboBoxEx_Destroy ' +
        'GUICtrlComboBoxEx_EndUpdate GUICtrlComboBoxEx_FindStringExact ' +
        'GUICtrlComboBoxEx_GetComboBoxInfo ' +
        'GUICtrlComboBoxEx_GetComboControl GUICtrlComboBoxEx_GetCount ' +
        'GUICtrlComboBoxEx_GetCurSel ' +
        'GUICtrlComboBoxEx_GetDroppedControlRect ' +
        'GUICtrlComboBoxEx_GetDroppedControlRectEx ' +
        'GUICtrlComboBoxEx_GetDroppedState ' +
        'GUICtrlComboBoxEx_GetDroppedWidth ' +
        'GUICtrlComboBoxEx_GetEditControl GUICtrlComboBoxEx_GetEditSel ' +
        'GUICtrlComboBoxEx_GetEditText ' +
        'GUICtrlComboBoxEx_GetExtendedStyle ' +
        'GUICtrlComboBoxEx_GetExtendedUI GUICtrlComboBoxEx_GetImageList ' +
        'GUICtrlComboBoxEx_GetItem GUICtrlComboBoxEx_GetItemEx ' +
        'GUICtrlComboBoxEx_GetItemHeight GUICtrlComboBoxEx_GetItemImage ' +
        'GUICtrlComboBoxEx_GetItemIndent ' +
        'GUICtrlComboBoxEx_GetItemOverlayImage ' +
        'GUICtrlComboBoxEx_GetItemParam ' +
        'GUICtrlComboBoxEx_GetItemSelectedImage ' +
        'GUICtrlComboBoxEx_GetItemText GUICtrlComboBoxEx_GetItemTextLen ' +
        'GUICtrlComboBoxEx_GetList GUICtrlComboBoxEx_GetListArray ' +
        'GUICtrlComboBoxEx_GetLocale GUICtrlComboBoxEx_GetLocaleCountry ' +
        'GUICtrlComboBoxEx_GetLocaleLang ' +
        'GUICtrlComboBoxEx_GetLocalePrimLang ' +
        'GUICtrlComboBoxEx_GetLocaleSubLang ' +
        'GUICtrlComboBoxEx_GetMinVisible GUICtrlComboBoxEx_GetTopIndex ' +
        'GUICtrlComboBoxEx_GetUnicode GUICtrlComboBoxEx_InitStorage ' +
        'GUICtrlComboBoxEx_InsertString GUICtrlComboBoxEx_LimitText ' +
        'GUICtrlComboBoxEx_ReplaceEditSel GUICtrlComboBoxEx_ResetContent ' +
        'GUICtrlComboBoxEx_SetCurSel GUICtrlComboBoxEx_SetDroppedWidth ' +
        'GUICtrlComboBoxEx_SetEditSel GUICtrlComboBoxEx_SetEditText ' +
        'GUICtrlComboBoxEx_SetExtendedStyle ' +
        'GUICtrlComboBoxEx_SetExtendedUI GUICtrlComboBoxEx_SetImageList ' +
        'GUICtrlComboBoxEx_SetItem GUICtrlComboBoxEx_SetItemEx ' +
        'GUICtrlComboBoxEx_SetItemHeight GUICtrlComboBoxEx_SetItemImage ' +
        'GUICtrlComboBoxEx_SetItemIndent ' +
        'GUICtrlComboBoxEx_SetItemOverlayImage ' +
        'GUICtrlComboBoxEx_SetItemParam ' +
        'GUICtrlComboBoxEx_SetItemSelectedImage ' +
        'GUICtrlComboBoxEx_SetMinVisible GUICtrlComboBoxEx_SetTopIndex ' +
        'GUICtrlComboBoxEx_SetUnicode GUICtrlComboBoxEx_ShowDropDown ' +
        'GUICtrlComboBox_AddDir GUICtrlComboBox_AddString ' +
        'GUICtrlComboBox_AutoComplete GUICtrlComboBox_BeginUpdate ' +
        'GUICtrlComboBox_Create GUICtrlComboBox_DeleteString ' +
        'GUICtrlComboBox_Destroy GUICtrlComboBox_EndUpdate ' +
        'GUICtrlComboBox_FindString GUICtrlComboBox_FindStringExact ' +
        'GUICtrlComboBox_GetComboBoxInfo GUICtrlComboBox_GetCount ' +
        'GUICtrlComboBox_GetCueBanner GUICtrlComboBox_GetCurSel ' +
        'GUICtrlComboBox_GetDroppedControlRect ' +
        'GUICtrlComboBox_GetDroppedControlRectEx ' +
        'GUICtrlComboBox_GetDroppedState GUICtrlComboBox_GetDroppedWidth ' +
        'GUICtrlComboBox_GetEditSel GUICtrlComboBox_GetEditText ' +
        'GUICtrlComboBox_GetExtendedUI ' +
        'GUICtrlComboBox_GetHorizontalExtent ' +
        'GUICtrlComboBox_GetItemHeight GUICtrlComboBox_GetLBText ' +
        'GUICtrlComboBox_GetLBTextLen GUICtrlComboBox_GetList ' +
        'GUICtrlComboBox_GetListArray GUICtrlComboBox_GetLocale ' +
        'GUICtrlComboBox_GetLocaleCountry GUICtrlComboBox_GetLocaleLang ' +
        'GUICtrlComboBox_GetLocalePrimLang ' +
        'GUICtrlComboBox_GetLocaleSubLang GUICtrlComboBox_GetMinVisible ' +
        'GUICtrlComboBox_GetTopIndex GUICtrlComboBox_InitStorage ' +
        'GUICtrlComboBox_InsertString GUICtrlComboBox_LimitText ' +
        'GUICtrlComboBox_ReplaceEditSel GUICtrlComboBox_ResetContent ' +
        'GUICtrlComboBox_SelectString GUICtrlComboBox_SetCueBanner ' +
        'GUICtrlComboBox_SetCurSel GUICtrlComboBox_SetDroppedWidth ' +
        'GUICtrlComboBox_SetEditSel GUICtrlComboBox_SetEditText ' +
        'GUICtrlComboBox_SetExtendedUI ' +
        'GUICtrlComboBox_SetHorizontalExtent ' +
        'GUICtrlComboBox_SetItemHeight GUICtrlComboBox_SetMinVisible ' +
        'GUICtrlComboBox_SetTopIndex GUICtrlComboBox_ShowDropDown ' +
        'GUICtrlDTP_Create GUICtrlDTP_Destroy GUICtrlDTP_GetMCColor ' +
        'GUICtrlDTP_GetMCFont GUICtrlDTP_GetMonthCal ' +
        'GUICtrlDTP_GetRange GUICtrlDTP_GetRangeEx ' +
        'GUICtrlDTP_GetSystemTime GUICtrlDTP_GetSystemTimeEx ' +
        'GUICtrlDTP_SetFormat GUICtrlDTP_SetMCColor ' +
        'GUICtrlDTP_SetMCFont GUICtrlDTP_SetRange ' +
        'GUICtrlDTP_SetRangeEx GUICtrlDTP_SetSystemTime ' +
        'GUICtrlDTP_SetSystemTimeEx GUICtrlEdit_AppendText ' +
        'GUICtrlEdit_BeginUpdate GUICtrlEdit_CanUndo ' +
        'GUICtrlEdit_CharFromPos GUICtrlEdit_Create ' +
        'GUICtrlEdit_Destroy GUICtrlEdit_EmptyUndoBuffer ' +
        'GUICtrlEdit_EndUpdate GUICtrlEdit_Find GUICtrlEdit_FmtLines ' +
        'GUICtrlEdit_GetCueBanner GUICtrlEdit_GetFirstVisibleLine ' +
        'GUICtrlEdit_GetLimitText GUICtrlEdit_GetLine ' +
        'GUICtrlEdit_GetLineCount GUICtrlEdit_GetMargins ' +
        'GUICtrlEdit_GetModify GUICtrlEdit_GetPasswordChar ' +
        'GUICtrlEdit_GetRECT GUICtrlEdit_GetRECTEx GUICtrlEdit_GetSel ' +
        'GUICtrlEdit_GetText GUICtrlEdit_GetTextLen ' +
        'GUICtrlEdit_HideBalloonTip GUICtrlEdit_InsertText ' +
        'GUICtrlEdit_LineFromChar GUICtrlEdit_LineIndex ' +
        'GUICtrlEdit_LineLength GUICtrlEdit_LineScroll ' +
        'GUICtrlEdit_PosFromChar GUICtrlEdit_ReplaceSel ' +
        'GUICtrlEdit_Scroll GUICtrlEdit_SetCueBanner ' +
        'GUICtrlEdit_SetLimitText GUICtrlEdit_SetMargins ' +
        'GUICtrlEdit_SetModify GUICtrlEdit_SetPasswordChar ' +
        'GUICtrlEdit_SetReadOnly GUICtrlEdit_SetRECT ' +
        'GUICtrlEdit_SetRECTEx GUICtrlEdit_SetRECTNP ' +
        'GUICtrlEdit_SetRectNPEx GUICtrlEdit_SetSel ' +
        'GUICtrlEdit_SetTabStops GUICtrlEdit_SetText ' +
        'GUICtrlEdit_ShowBalloonTip GUICtrlEdit_Undo ' +
        'GUICtrlHeader_AddItem GUICtrlHeader_ClearFilter ' +
        'GUICtrlHeader_ClearFilterAll GUICtrlHeader_Create ' +
        'GUICtrlHeader_CreateDragImage GUICtrlHeader_DeleteItem ' +
        'GUICtrlHeader_Destroy GUICtrlHeader_EditFilter ' +
        'GUICtrlHeader_GetBitmapMargin GUICtrlHeader_GetImageList ' +
        'GUICtrlHeader_GetItem GUICtrlHeader_GetItemAlign ' +
        'GUICtrlHeader_GetItemBitmap GUICtrlHeader_GetItemCount ' +
        'GUICtrlHeader_GetItemDisplay GUICtrlHeader_GetItemFlags ' +
        'GUICtrlHeader_GetItemFormat GUICtrlHeader_GetItemImage ' +
        'GUICtrlHeader_GetItemOrder GUICtrlHeader_GetItemParam ' +
        'GUICtrlHeader_GetItemRect GUICtrlHeader_GetItemRectEx ' +
        'GUICtrlHeader_GetItemText GUICtrlHeader_GetItemWidth ' +
        'GUICtrlHeader_GetOrderArray GUICtrlHeader_GetUnicodeFormat ' +
        'GUICtrlHeader_HitTest GUICtrlHeader_InsertItem ' +
        'GUICtrlHeader_Layout GUICtrlHeader_OrderToIndex ' +
        'GUICtrlHeader_SetBitmapMargin ' +
        'GUICtrlHeader_SetFilterChangeTimeout ' +
        'GUICtrlHeader_SetHotDivider GUICtrlHeader_SetImageList ' +
        'GUICtrlHeader_SetItem GUICtrlHeader_SetItemAlign ' +
        'GUICtrlHeader_SetItemBitmap GUICtrlHeader_SetItemDisplay ' +
        'GUICtrlHeader_SetItemFlags GUICtrlHeader_SetItemFormat ' +
        'GUICtrlHeader_SetItemImage GUICtrlHeader_SetItemOrder ' +
        'GUICtrlHeader_SetItemParam GUICtrlHeader_SetItemText ' +
        'GUICtrlHeader_SetItemWidth GUICtrlHeader_SetOrderArray ' +
        'GUICtrlHeader_SetUnicodeFormat GUICtrlIpAddress_ClearAddress ' +
        'GUICtrlIpAddress_Create GUICtrlIpAddress_Destroy ' +
        'GUICtrlIpAddress_Get GUICtrlIpAddress_GetArray ' +
        'GUICtrlIpAddress_GetEx GUICtrlIpAddress_IsBlank ' +
        'GUICtrlIpAddress_Set GUICtrlIpAddress_SetArray ' +
        'GUICtrlIpAddress_SetEx GUICtrlIpAddress_SetFocus ' +
        'GUICtrlIpAddress_SetFont GUICtrlIpAddress_SetRange ' +
        'GUICtrlIpAddress_ShowHide GUICtrlListBox_AddFile ' +
        'GUICtrlListBox_AddString GUICtrlListBox_BeginUpdate ' +
        'GUICtrlListBox_ClickItem GUICtrlListBox_Create ' +
        'GUICtrlListBox_DeleteString GUICtrlListBox_Destroy ' +
        'GUICtrlListBox_Dir GUICtrlListBox_EndUpdate ' +
        'GUICtrlListBox_FindInText GUICtrlListBox_FindString ' +
        'GUICtrlListBox_GetAnchorIndex GUICtrlListBox_GetCaretIndex ' +
        'GUICtrlListBox_GetCount GUICtrlListBox_GetCurSel ' +
        'GUICtrlListBox_GetHorizontalExtent GUICtrlListBox_GetItemData ' +
        'GUICtrlListBox_GetItemHeight GUICtrlListBox_GetItemRect ' +
        'GUICtrlListBox_GetItemRectEx GUICtrlListBox_GetListBoxInfo ' +
        'GUICtrlListBox_GetLocale GUICtrlListBox_GetLocaleCountry ' +
        'GUICtrlListBox_GetLocaleLang GUICtrlListBox_GetLocalePrimLang ' +
        'GUICtrlListBox_GetLocaleSubLang GUICtrlListBox_GetSel ' +
        'GUICtrlListBox_GetSelCount GUICtrlListBox_GetSelItems ' +
        'GUICtrlListBox_GetSelItemsText GUICtrlListBox_GetText ' +
        'GUICtrlListBox_GetTextLen GUICtrlListBox_GetTopIndex ' +
        'GUICtrlListBox_InitStorage GUICtrlListBox_InsertString ' +
        'GUICtrlListBox_ItemFromPoint GUICtrlListBox_ReplaceString ' +
        'GUICtrlListBox_ResetContent GUICtrlListBox_SelectString ' +
        'GUICtrlListBox_SelItemRange GUICtrlListBox_SelItemRangeEx ' +
        'GUICtrlListBox_SetAnchorIndex GUICtrlListBox_SetCaretIndex ' +
        'GUICtrlListBox_SetColumnWidth GUICtrlListBox_SetCurSel ' +
        'GUICtrlListBox_SetHorizontalExtent GUICtrlListBox_SetItemData ' +
        'GUICtrlListBox_SetItemHeight GUICtrlListBox_SetLocale ' +
        'GUICtrlListBox_SetSel GUICtrlListBox_SetTabStops ' +
        'GUICtrlListBox_SetTopIndex GUICtrlListBox_Sort ' +
        'GUICtrlListBox_SwapString GUICtrlListBox_UpdateHScroll ' +
        'GUICtrlListView_AddArray GUICtrlListView_AddColumn ' +
        'GUICtrlListView_AddItem GUICtrlListView_AddSubItem ' +
        'GUICtrlListView_ApproximateViewHeight ' +
        'GUICtrlListView_ApproximateViewRect ' +
        'GUICtrlListView_ApproximateViewWidth GUICtrlListView_Arrange ' +
        'GUICtrlListView_BeginUpdate GUICtrlListView_CancelEditLabel ' +
        'GUICtrlListView_ClickItem GUICtrlListView_CopyItems ' +
        'GUICtrlListView_Create GUICtrlListView_CreateDragImage ' +
        'GUICtrlListView_CreateSolidBitMap ' +
        'GUICtrlListView_DeleteAllItems GUICtrlListView_DeleteColumn ' +
        'GUICtrlListView_DeleteItem GUICtrlListView_DeleteItemsSelected ' +
        'GUICtrlListView_Destroy GUICtrlListView_DrawDragImage ' +
        'GUICtrlListView_EditLabel GUICtrlListView_EnableGroupView ' +
        'GUICtrlListView_EndUpdate GUICtrlListView_EnsureVisible ' +
        'GUICtrlListView_FindInText GUICtrlListView_FindItem ' +
        'GUICtrlListView_FindNearest GUICtrlListView_FindParam ' +
        'GUICtrlListView_FindText GUICtrlListView_GetBkColor ' +
        'GUICtrlListView_GetBkImage GUICtrlListView_GetCallbackMask ' +
        'GUICtrlListView_GetColumn GUICtrlListView_GetColumnCount ' +
        'GUICtrlListView_GetColumnOrder ' +
        'GUICtrlListView_GetColumnOrderArray ' +
        'GUICtrlListView_GetColumnWidth GUICtrlListView_GetCounterPage ' +
        'GUICtrlListView_GetEditControl ' +
        'GUICtrlListView_GetExtendedListViewStyle ' +
        'GUICtrlListView_GetFocusedGroup GUICtrlListView_GetGroupCount ' +
        'GUICtrlListView_GetGroupInfo ' +
        'GUICtrlListView_GetGroupInfoByIndex ' +
        'GUICtrlListView_GetGroupRect ' +
        'GUICtrlListView_GetGroupViewEnabled GUICtrlListView_GetHeader ' +
        'GUICtrlListView_GetHotCursor GUICtrlListView_GetHotItem ' +
        'GUICtrlListView_GetHoverTime GUICtrlListView_GetImageList ' +
        'GUICtrlListView_GetISearchString GUICtrlListView_GetItem ' +
        'GUICtrlListView_GetItemChecked GUICtrlListView_GetItemCount ' +
        'GUICtrlListView_GetItemCut GUICtrlListView_GetItemDropHilited ' +
        'GUICtrlListView_GetItemEx GUICtrlListView_GetItemFocused ' +
        'GUICtrlListView_GetItemGroupID GUICtrlListView_GetItemImage ' +
        'GUICtrlListView_GetItemIndent GUICtrlListView_GetItemParam ' +
        'GUICtrlListView_GetItemPosition ' +
        'GUICtrlListView_GetItemPositionX ' +
        'GUICtrlListView_GetItemPositionY GUICtrlListView_GetItemRect ' +
        'GUICtrlListView_GetItemRectEx GUICtrlListView_GetItemSelected ' +
        'GUICtrlListView_GetItemSpacing GUICtrlListView_GetItemSpacingX ' +
        'GUICtrlListView_GetItemSpacingY GUICtrlListView_GetItemState ' +
        'GUICtrlListView_GetItemStateImage GUICtrlListView_GetItemText ' +
        'GUICtrlListView_GetItemTextArray ' +
        'GUICtrlListView_GetItemTextString GUICtrlListView_GetNextItem ' +
        'GUICtrlListView_GetNumberOfWorkAreas GUICtrlListView_GetOrigin ' +
        'GUICtrlListView_GetOriginX GUICtrlListView_GetOriginY ' +
        'GUICtrlListView_GetOutlineColor ' +
        'GUICtrlListView_GetSelectedColumn ' +
        'GUICtrlListView_GetSelectedCount ' +
        'GUICtrlListView_GetSelectedIndices ' +
        'GUICtrlListView_GetSelectionMark GUICtrlListView_GetStringWidth ' +
        'GUICtrlListView_GetSubItemRect GUICtrlListView_GetTextBkColor ' +
        'GUICtrlListView_GetTextColor GUICtrlListView_GetToolTips ' +
        'GUICtrlListView_GetTopIndex GUICtrlListView_GetUnicodeFormat ' +
        'GUICtrlListView_GetView GUICtrlListView_GetViewDetails ' +
        'GUICtrlListView_GetViewLarge GUICtrlListView_GetViewList ' +
        'GUICtrlListView_GetViewRect GUICtrlListView_GetViewSmall ' +
        'GUICtrlListView_GetViewTile GUICtrlListView_HideColumn ' +
        'GUICtrlListView_HitTest GUICtrlListView_InsertColumn ' +
        'GUICtrlListView_InsertGroup GUICtrlListView_InsertItem ' +
        'GUICtrlListView_JustifyColumn GUICtrlListView_MapIDToIndex ' +
        'GUICtrlListView_MapIndexToID GUICtrlListView_RedrawItems ' +
        'GUICtrlListView_RegisterSortCallBack ' +
        'GUICtrlListView_RemoveAllGroups GUICtrlListView_RemoveGroup ' +
        'GUICtrlListView_Scroll GUICtrlListView_SetBkColor ' +
        'GUICtrlListView_SetBkImage GUICtrlListView_SetCallBackMask ' +
        'GUICtrlListView_SetColumn GUICtrlListView_SetColumnOrder ' +
        'GUICtrlListView_SetColumnOrderArray ' +
        'GUICtrlListView_SetColumnWidth ' +
        'GUICtrlListView_SetExtendedListViewStyle ' +
        'GUICtrlListView_SetGroupInfo GUICtrlListView_SetHotItem ' +
        'GUICtrlListView_SetHoverTime GUICtrlListView_SetIconSpacing ' +
        'GUICtrlListView_SetImageList GUICtrlListView_SetItem ' +
        'GUICtrlListView_SetItemChecked GUICtrlListView_SetItemCount ' +
        'GUICtrlListView_SetItemCut GUICtrlListView_SetItemDropHilited ' +
        'GUICtrlListView_SetItemEx GUICtrlListView_SetItemFocused ' +
        'GUICtrlListView_SetItemGroupID GUICtrlListView_SetItemImage ' +
        'GUICtrlListView_SetItemIndent GUICtrlListView_SetItemParam ' +
        'GUICtrlListView_SetItemPosition ' +
        'GUICtrlListView_SetItemPosition32 ' +
        'GUICtrlListView_SetItemSelected GUICtrlListView_SetItemState ' +
        'GUICtrlListView_SetItemStateImage GUICtrlListView_SetItemText ' +
        'GUICtrlListView_SetOutlineColor ' +
        'GUICtrlListView_SetSelectedColumn ' +
        'GUICtrlListView_SetSelectionMark GUICtrlListView_SetTextBkColor ' +
        'GUICtrlListView_SetTextColor GUICtrlListView_SetToolTips ' +
        'GUICtrlListView_SetUnicodeFormat GUICtrlListView_SetView ' +
        'GUICtrlListView_SetWorkAreas GUICtrlListView_SimpleSort ' +
        'GUICtrlListView_SortItems GUICtrlListView_SubItemHitTest ' +
        'GUICtrlListView_UnRegisterSortCallBack GUICtrlMenu_AddMenuItem ' +
        'GUICtrlMenu_AppendMenu GUICtrlMenu_CalculatePopupWindowPosition ' +
        'GUICtrlMenu_CheckMenuItem GUICtrlMenu_CheckRadioItem ' +
        'GUICtrlMenu_CreateMenu GUICtrlMenu_CreatePopup ' +
        'GUICtrlMenu_DeleteMenu GUICtrlMenu_DestroyMenu ' +
        'GUICtrlMenu_DrawMenuBar GUICtrlMenu_EnableMenuItem ' +
        'GUICtrlMenu_FindItem GUICtrlMenu_FindParent ' +
        'GUICtrlMenu_GetItemBmp GUICtrlMenu_GetItemBmpChecked ' +
        'GUICtrlMenu_GetItemBmpUnchecked GUICtrlMenu_GetItemChecked ' +
        'GUICtrlMenu_GetItemCount GUICtrlMenu_GetItemData ' +
        'GUICtrlMenu_GetItemDefault GUICtrlMenu_GetItemDisabled ' +
        'GUICtrlMenu_GetItemEnabled GUICtrlMenu_GetItemGrayed ' +
        'GUICtrlMenu_GetItemHighlighted GUICtrlMenu_GetItemID ' +
        'GUICtrlMenu_GetItemInfo GUICtrlMenu_GetItemRect ' +
        'GUICtrlMenu_GetItemRectEx GUICtrlMenu_GetItemState ' +
        'GUICtrlMenu_GetItemStateEx GUICtrlMenu_GetItemSubMenu ' +
        'GUICtrlMenu_GetItemText GUICtrlMenu_GetItemType ' +
        'GUICtrlMenu_GetMenu GUICtrlMenu_GetMenuBackground ' +
        'GUICtrlMenu_GetMenuBarInfo GUICtrlMenu_GetMenuContextHelpID ' +
        'GUICtrlMenu_GetMenuData GUICtrlMenu_GetMenuDefaultItem ' +
        'GUICtrlMenu_GetMenuHeight GUICtrlMenu_GetMenuInfo ' +
        'GUICtrlMenu_GetMenuStyle GUICtrlMenu_GetSystemMenu ' +
        'GUICtrlMenu_InsertMenuItem GUICtrlMenu_InsertMenuItemEx ' +
        'GUICtrlMenu_IsMenu GUICtrlMenu_LoadMenu ' +
        'GUICtrlMenu_MapAccelerator GUICtrlMenu_MenuItemFromPoint ' +
        'GUICtrlMenu_RemoveMenu GUICtrlMenu_SetItemBitmaps ' +
        'GUICtrlMenu_SetItemBmp GUICtrlMenu_SetItemBmpChecked ' +
        'GUICtrlMenu_SetItemBmpUnchecked GUICtrlMenu_SetItemChecked ' +
        'GUICtrlMenu_SetItemData GUICtrlMenu_SetItemDefault ' +
        'GUICtrlMenu_SetItemDisabled GUICtrlMenu_SetItemEnabled ' +
        'GUICtrlMenu_SetItemGrayed GUICtrlMenu_SetItemHighlighted ' +
        'GUICtrlMenu_SetItemID GUICtrlMenu_SetItemInfo ' +
        'GUICtrlMenu_SetItemState GUICtrlMenu_SetItemSubMenu ' +
        'GUICtrlMenu_SetItemText GUICtrlMenu_SetItemType ' +
        'GUICtrlMenu_SetMenu GUICtrlMenu_SetMenuBackground ' +
        'GUICtrlMenu_SetMenuContextHelpID GUICtrlMenu_SetMenuData ' +
        'GUICtrlMenu_SetMenuDefaultItem GUICtrlMenu_SetMenuHeight ' +
        'GUICtrlMenu_SetMenuInfo GUICtrlMenu_SetMenuStyle ' +
        'GUICtrlMenu_TrackPopupMenu GUICtrlMonthCal_Create ' +
        'GUICtrlMonthCal_Destroy GUICtrlMonthCal_GetCalendarBorder ' +
        'GUICtrlMonthCal_GetCalendarCount GUICtrlMonthCal_GetColor ' +
        'GUICtrlMonthCal_GetColorArray GUICtrlMonthCal_GetCurSel ' +
        'GUICtrlMonthCal_GetCurSelStr GUICtrlMonthCal_GetFirstDOW ' +
        'GUICtrlMonthCal_GetFirstDOWStr GUICtrlMonthCal_GetMaxSelCount ' +
        'GUICtrlMonthCal_GetMaxTodayWidth ' +
        'GUICtrlMonthCal_GetMinReqHeight GUICtrlMonthCal_GetMinReqRect ' +
        'GUICtrlMonthCal_GetMinReqRectArray ' +
        'GUICtrlMonthCal_GetMinReqWidth GUICtrlMonthCal_GetMonthDelta ' +
        'GUICtrlMonthCal_GetMonthRange GUICtrlMonthCal_GetMonthRangeMax ' +
        'GUICtrlMonthCal_GetMonthRangeMaxStr ' +
        'GUICtrlMonthCal_GetMonthRangeMin ' +
        'GUICtrlMonthCal_GetMonthRangeMinStr ' +
        'GUICtrlMonthCal_GetMonthRangeSpan GUICtrlMonthCal_GetRange ' +
        'GUICtrlMonthCal_GetRangeMax GUICtrlMonthCal_GetRangeMaxStr ' +
        'GUICtrlMonthCal_GetRangeMin GUICtrlMonthCal_GetRangeMinStr ' +
        'GUICtrlMonthCal_GetSelRange GUICtrlMonthCal_GetSelRangeMax ' +
        'GUICtrlMonthCal_GetSelRangeMaxStr ' +
        'GUICtrlMonthCal_GetSelRangeMin ' +
        'GUICtrlMonthCal_GetSelRangeMinStr GUICtrlMonthCal_GetToday ' +
        'GUICtrlMonthCal_GetTodayStr GUICtrlMonthCal_GetUnicodeFormat ' +
        'GUICtrlMonthCal_HitTest GUICtrlMonthCal_SetCalendarBorder ' +
        'GUICtrlMonthCal_SetColor GUICtrlMonthCal_SetCurSel ' +
        'GUICtrlMonthCal_SetDayState GUICtrlMonthCal_SetFirstDOW ' +
        'GUICtrlMonthCal_SetMaxSelCount GUICtrlMonthCal_SetMonthDelta ' +
        'GUICtrlMonthCal_SetRange GUICtrlMonthCal_SetSelRange ' +
        'GUICtrlMonthCal_SetToday GUICtrlMonthCal_SetUnicodeFormat ' +
        'GUICtrlRebar_AddBand GUICtrlRebar_AddToolBarBand ' +
        'GUICtrlRebar_BeginDrag GUICtrlRebar_Create ' +
        'GUICtrlRebar_DeleteBand GUICtrlRebar_Destroy ' +
        'GUICtrlRebar_DragMove GUICtrlRebar_EndDrag ' +
        'GUICtrlRebar_GetBandBackColor GUICtrlRebar_GetBandBorders ' +
        'GUICtrlRebar_GetBandBordersEx GUICtrlRebar_GetBandChildHandle ' +
        'GUICtrlRebar_GetBandChildSize GUICtrlRebar_GetBandCount ' +
        'GUICtrlRebar_GetBandForeColor GUICtrlRebar_GetBandHeaderSize ' +
        'GUICtrlRebar_GetBandID GUICtrlRebar_GetBandIdealSize ' +
        'GUICtrlRebar_GetBandLength GUICtrlRebar_GetBandLParam ' +
        'GUICtrlRebar_GetBandMargins GUICtrlRebar_GetBandMarginsEx ' +
        'GUICtrlRebar_GetBandRect GUICtrlRebar_GetBandRectEx ' +
        'GUICtrlRebar_GetBandStyle GUICtrlRebar_GetBandStyleBreak ' +
        'GUICtrlRebar_GetBandStyleChildEdge ' +
        'GUICtrlRebar_GetBandStyleFixedBMP ' +
        'GUICtrlRebar_GetBandStyleFixedSize ' +
        'GUICtrlRebar_GetBandStyleGripperAlways ' +
        'GUICtrlRebar_GetBandStyleHidden ' +
        'GUICtrlRebar_GetBandStyleHideTitle ' +
        'GUICtrlRebar_GetBandStyleNoGripper ' +
        'GUICtrlRebar_GetBandStyleTopAlign ' +
        'GUICtrlRebar_GetBandStyleUseChevron ' +
        'GUICtrlRebar_GetBandStyleVariableHeight ' +
        'GUICtrlRebar_GetBandText GUICtrlRebar_GetBarHeight ' +
        'GUICtrlRebar_GetBarInfo GUICtrlRebar_GetBKColor ' +
        'GUICtrlRebar_GetColorScheme GUICtrlRebar_GetRowCount ' +
        'GUICtrlRebar_GetRowHeight GUICtrlRebar_GetTextColor ' +
        'GUICtrlRebar_GetToolTips GUICtrlRebar_GetUnicodeFormat ' +
        'GUICtrlRebar_HitTest GUICtrlRebar_IDToIndex ' +
        'GUICtrlRebar_MaximizeBand GUICtrlRebar_MinimizeBand ' +
        'GUICtrlRebar_MoveBand GUICtrlRebar_SetBandBackColor ' +
        'GUICtrlRebar_SetBandForeColor GUICtrlRebar_SetBandHeaderSize ' +
        'GUICtrlRebar_SetBandID GUICtrlRebar_SetBandIdealSize ' +
        'GUICtrlRebar_SetBandLength GUICtrlRebar_SetBandLParam ' +
        'GUICtrlRebar_SetBandStyle GUICtrlRebar_SetBandStyleBreak ' +
        'GUICtrlRebar_SetBandStyleChildEdge ' +
        'GUICtrlRebar_SetBandStyleFixedBMP ' +
        'GUICtrlRebar_SetBandStyleFixedSize ' +
        'GUICtrlRebar_SetBandStyleGripperAlways ' +
        'GUICtrlRebar_SetBandStyleHidden ' +
        'GUICtrlRebar_SetBandStyleHideTitle ' +
        'GUICtrlRebar_SetBandStyleNoGripper ' +
        'GUICtrlRebar_SetBandStyleTopAlign ' +
        'GUICtrlRebar_SetBandStyleUseChevron ' +
        'GUICtrlRebar_SetBandStyleVariableHeight ' +
        'GUICtrlRebar_SetBandText GUICtrlRebar_SetBarInfo ' +
        'GUICtrlRebar_SetBKColor GUICtrlRebar_SetColorScheme ' +
        'GUICtrlRebar_SetTextColor GUICtrlRebar_SetToolTips ' +
        'GUICtrlRebar_SetUnicodeFormat GUICtrlRebar_ShowBand ' +
        'GUICtrlRichEdit_AppendText GUICtrlRichEdit_AutoDetectURL ' +
        'GUICtrlRichEdit_CanPaste GUICtrlRichEdit_CanPasteSpecial ' +
        'GUICtrlRichEdit_CanRedo GUICtrlRichEdit_CanUndo ' +
        'GUICtrlRichEdit_ChangeFontSize GUICtrlRichEdit_Copy ' +
        'GUICtrlRichEdit_Create GUICtrlRichEdit_Cut ' +
        'GUICtrlRichEdit_Deselect GUICtrlRichEdit_Destroy ' +
        'GUICtrlRichEdit_EmptyUndoBuffer GUICtrlRichEdit_FindText ' +
        'GUICtrlRichEdit_FindTextInRange GUICtrlRichEdit_GetBkColor ' +
        'GUICtrlRichEdit_GetCharAttributes ' +
        'GUICtrlRichEdit_GetCharBkColor GUICtrlRichEdit_GetCharColor ' +
        'GUICtrlRichEdit_GetCharPosFromXY ' +
        'GUICtrlRichEdit_GetCharPosOfNextWord ' +
        'GUICtrlRichEdit_GetCharPosOfPreviousWord ' +
        'GUICtrlRichEdit_GetCharWordBreakInfo ' +
        'GUICtrlRichEdit_GetFirstCharPosOnLine GUICtrlRichEdit_GetFont ' +
        'GUICtrlRichEdit_GetLineCount GUICtrlRichEdit_GetLineLength ' +
        'GUICtrlRichEdit_GetLineNumberFromCharPos ' +
        'GUICtrlRichEdit_GetNextRedo GUICtrlRichEdit_GetNextUndo ' +
        'GUICtrlRichEdit_GetNumberOfFirstVisibleLine ' +
        'GUICtrlRichEdit_GetParaAlignment ' +
        'GUICtrlRichEdit_GetParaAttributes GUICtrlRichEdit_GetParaBorder ' +
        'GUICtrlRichEdit_GetParaIndents GUICtrlRichEdit_GetParaNumbering ' +
        'GUICtrlRichEdit_GetParaShading GUICtrlRichEdit_GetParaSpacing ' +
        'GUICtrlRichEdit_GetParaTabStops GUICtrlRichEdit_GetPasswordChar ' +
        'GUICtrlRichEdit_GetRECT GUICtrlRichEdit_GetScrollPos ' +
        'GUICtrlRichEdit_GetSel GUICtrlRichEdit_GetSelAA ' +
        'GUICtrlRichEdit_GetSelText GUICtrlRichEdit_GetSpaceUnit ' +
        'GUICtrlRichEdit_GetText GUICtrlRichEdit_GetTextInLine ' +
        'GUICtrlRichEdit_GetTextInRange GUICtrlRichEdit_GetTextLength ' +
        'GUICtrlRichEdit_GetVersion GUICtrlRichEdit_GetXYFromCharPos ' +
        'GUICtrlRichEdit_GetZoom GUICtrlRichEdit_GotoCharPos ' +
        'GUICtrlRichEdit_HideSelection GUICtrlRichEdit_InsertText ' +
        'GUICtrlRichEdit_IsModified GUICtrlRichEdit_IsTextSelected ' +
        'GUICtrlRichEdit_Paste GUICtrlRichEdit_PasteSpecial ' +
        'GUICtrlRichEdit_PauseRedraw GUICtrlRichEdit_Redo ' +
        'GUICtrlRichEdit_ReplaceText GUICtrlRichEdit_ResumeRedraw ' +
        'GUICtrlRichEdit_ScrollLineOrPage GUICtrlRichEdit_ScrollLines ' +
        'GUICtrlRichEdit_ScrollToCaret GUICtrlRichEdit_SetBkColor ' +
        'GUICtrlRichEdit_SetCharAttributes ' +
        'GUICtrlRichEdit_SetCharBkColor GUICtrlRichEdit_SetCharColor ' +
        'GUICtrlRichEdit_SetEventMask GUICtrlRichEdit_SetFont ' +
        'GUICtrlRichEdit_SetLimitOnText GUICtrlRichEdit_SetModified ' +
        'GUICtrlRichEdit_SetParaAlignment ' +
        'GUICtrlRichEdit_SetParaAttributes GUICtrlRichEdit_SetParaBorder ' +
        'GUICtrlRichEdit_SetParaIndents GUICtrlRichEdit_SetParaNumbering ' +
        'GUICtrlRichEdit_SetParaShading GUICtrlRichEdit_SetParaSpacing ' +
        'GUICtrlRichEdit_SetParaTabStops GUICtrlRichEdit_SetPasswordChar ' +
        'GUICtrlRichEdit_SetReadOnly GUICtrlRichEdit_SetRECT ' +
        'GUICtrlRichEdit_SetScrollPos GUICtrlRichEdit_SetSel ' +
        'GUICtrlRichEdit_SetSpaceUnit GUICtrlRichEdit_SetTabStops ' +
        'GUICtrlRichEdit_SetText GUICtrlRichEdit_SetUndoLimit ' +
        'GUICtrlRichEdit_SetZoom GUICtrlRichEdit_StreamFromFile ' +
        'GUICtrlRichEdit_StreamFromVar GUICtrlRichEdit_StreamToFile ' +
        'GUICtrlRichEdit_StreamToVar GUICtrlRichEdit_Undo ' +
        'GUICtrlSlider_ClearSel GUICtrlSlider_ClearTics ' +
        'GUICtrlSlider_Create GUICtrlSlider_Destroy ' +
        'GUICtrlSlider_GetBuddy GUICtrlSlider_GetChannelRect ' +
        'GUICtrlSlider_GetChannelRectEx GUICtrlSlider_GetLineSize ' +
        'GUICtrlSlider_GetLogicalTics GUICtrlSlider_GetNumTics ' +
        'GUICtrlSlider_GetPageSize GUICtrlSlider_GetPos ' +
        'GUICtrlSlider_GetRange GUICtrlSlider_GetRangeMax ' +
        'GUICtrlSlider_GetRangeMin GUICtrlSlider_GetSel ' +
        'GUICtrlSlider_GetSelEnd GUICtrlSlider_GetSelStart ' +
        'GUICtrlSlider_GetThumbLength GUICtrlSlider_GetThumbRect ' +
        'GUICtrlSlider_GetThumbRectEx GUICtrlSlider_GetTic ' +
        'GUICtrlSlider_GetTicPos GUICtrlSlider_GetToolTips ' +
        'GUICtrlSlider_GetUnicodeFormat GUICtrlSlider_SetBuddy ' +
        'GUICtrlSlider_SetLineSize GUICtrlSlider_SetPageSize ' +
        'GUICtrlSlider_SetPos GUICtrlSlider_SetRange ' +
        'GUICtrlSlider_SetRangeMax GUICtrlSlider_SetRangeMin ' +
        'GUICtrlSlider_SetSel GUICtrlSlider_SetSelEnd ' +
        'GUICtrlSlider_SetSelStart GUICtrlSlider_SetThumbLength ' +
        'GUICtrlSlider_SetTic GUICtrlSlider_SetTicFreq ' +
        'GUICtrlSlider_SetTipSide GUICtrlSlider_SetToolTips ' +
        'GUICtrlSlider_SetUnicodeFormat GUICtrlStatusBar_Create ' +
        'GUICtrlStatusBar_Destroy GUICtrlStatusBar_EmbedControl ' +
        'GUICtrlStatusBar_GetBorders GUICtrlStatusBar_GetBordersHorz ' +
        'GUICtrlStatusBar_GetBordersRect GUICtrlStatusBar_GetBordersVert ' +
        'GUICtrlStatusBar_GetCount GUICtrlStatusBar_GetHeight ' +
        'GUICtrlStatusBar_GetIcon GUICtrlStatusBar_GetParts ' +
        'GUICtrlStatusBar_GetRect GUICtrlStatusBar_GetRectEx ' +
        'GUICtrlStatusBar_GetText GUICtrlStatusBar_GetTextFlags ' +
        'GUICtrlStatusBar_GetTextLength GUICtrlStatusBar_GetTextLengthEx ' +
        'GUICtrlStatusBar_GetTipText GUICtrlStatusBar_GetUnicodeFormat ' +
        'GUICtrlStatusBar_GetWidth GUICtrlStatusBar_IsSimple ' +
        'GUICtrlStatusBar_Resize GUICtrlStatusBar_SetBkColor ' +
        'GUICtrlStatusBar_SetIcon GUICtrlStatusBar_SetMinHeight ' +
        'GUICtrlStatusBar_SetParts GUICtrlStatusBar_SetSimple ' +
        'GUICtrlStatusBar_SetText GUICtrlStatusBar_SetTipText ' +
        'GUICtrlStatusBar_SetUnicodeFormat GUICtrlStatusBar_ShowHide ' +
        'GUICtrlTab_ActivateTab GUICtrlTab_ClickTab GUICtrlTab_Create ' +
        'GUICtrlTab_DeleteAllItems GUICtrlTab_DeleteItem ' +
        'GUICtrlTab_DeselectAll GUICtrlTab_Destroy GUICtrlTab_FindTab ' +
        'GUICtrlTab_GetCurFocus GUICtrlTab_GetCurSel ' +
        'GUICtrlTab_GetDisplayRect GUICtrlTab_GetDisplayRectEx ' +
        'GUICtrlTab_GetExtendedStyle GUICtrlTab_GetImageList ' +
        'GUICtrlTab_GetItem GUICtrlTab_GetItemCount ' +
        'GUICtrlTab_GetItemImage GUICtrlTab_GetItemParam ' +
        'GUICtrlTab_GetItemRect GUICtrlTab_GetItemRectEx ' +
        'GUICtrlTab_GetItemState GUICtrlTab_GetItemText ' +
        'GUICtrlTab_GetRowCount GUICtrlTab_GetToolTips ' +
        'GUICtrlTab_GetUnicodeFormat GUICtrlTab_HighlightItem ' +
        'GUICtrlTab_HitTest GUICtrlTab_InsertItem ' +
        'GUICtrlTab_RemoveImage GUICtrlTab_SetCurFocus ' +
        'GUICtrlTab_SetCurSel GUICtrlTab_SetExtendedStyle ' +
        'GUICtrlTab_SetImageList GUICtrlTab_SetItem ' +
        'GUICtrlTab_SetItemImage GUICtrlTab_SetItemParam ' +
        'GUICtrlTab_SetItemSize GUICtrlTab_SetItemState ' +
        'GUICtrlTab_SetItemText GUICtrlTab_SetMinTabWidth ' +
        'GUICtrlTab_SetPadding GUICtrlTab_SetToolTips ' +
        'GUICtrlTab_SetUnicodeFormat GUICtrlToolbar_AddBitmap ' +
        'GUICtrlToolbar_AddButton GUICtrlToolbar_AddButtonSep ' +
        'GUICtrlToolbar_AddString GUICtrlToolbar_ButtonCount ' +
        'GUICtrlToolbar_CheckButton GUICtrlToolbar_ClickAccel ' +
        'GUICtrlToolbar_ClickButton GUICtrlToolbar_ClickIndex ' +
        'GUICtrlToolbar_CommandToIndex GUICtrlToolbar_Create ' +
        'GUICtrlToolbar_Customize GUICtrlToolbar_DeleteButton ' +
        'GUICtrlToolbar_Destroy GUICtrlToolbar_EnableButton ' +
        'GUICtrlToolbar_FindToolbar GUICtrlToolbar_GetAnchorHighlight ' +
        'GUICtrlToolbar_GetBitmapFlags GUICtrlToolbar_GetButtonBitmap ' +
        'GUICtrlToolbar_GetButtonInfo GUICtrlToolbar_GetButtonInfoEx ' +
        'GUICtrlToolbar_GetButtonParam GUICtrlToolbar_GetButtonRect ' +
        'GUICtrlToolbar_GetButtonRectEx GUICtrlToolbar_GetButtonSize ' +
        'GUICtrlToolbar_GetButtonState GUICtrlToolbar_GetButtonStyle ' +
        'GUICtrlToolbar_GetButtonText GUICtrlToolbar_GetColorScheme ' +
        'GUICtrlToolbar_GetDisabledImageList ' +
        'GUICtrlToolbar_GetExtendedStyle GUICtrlToolbar_GetHotImageList ' +
        'GUICtrlToolbar_GetHotItem GUICtrlToolbar_GetImageList ' +
        'GUICtrlToolbar_GetInsertMark GUICtrlToolbar_GetInsertMarkColor ' +
        'GUICtrlToolbar_GetMaxSize GUICtrlToolbar_GetMetrics ' +
        'GUICtrlToolbar_GetPadding GUICtrlToolbar_GetRows ' +
        'GUICtrlToolbar_GetString GUICtrlToolbar_GetStyle ' +
        'GUICtrlToolbar_GetStyleAltDrag ' +
        'GUICtrlToolbar_GetStyleCustomErase GUICtrlToolbar_GetStyleFlat ' +
        'GUICtrlToolbar_GetStyleList GUICtrlToolbar_GetStyleRegisterDrop ' +
        'GUICtrlToolbar_GetStyleToolTips ' +
        'GUICtrlToolbar_GetStyleTransparent ' +
        'GUICtrlToolbar_GetStyleWrapable GUICtrlToolbar_GetTextRows ' +
        'GUICtrlToolbar_GetToolTips GUICtrlToolbar_GetUnicodeFormat ' +
        'GUICtrlToolbar_HideButton GUICtrlToolbar_HighlightButton ' +
        'GUICtrlToolbar_HitTest GUICtrlToolbar_IndexToCommand ' +
        'GUICtrlToolbar_InsertButton GUICtrlToolbar_InsertMarkHitTest ' +
        'GUICtrlToolbar_IsButtonChecked GUICtrlToolbar_IsButtonEnabled ' +
        'GUICtrlToolbar_IsButtonHidden ' +
        'GUICtrlToolbar_IsButtonHighlighted ' +
        'GUICtrlToolbar_IsButtonIndeterminate ' +
        'GUICtrlToolbar_IsButtonPressed GUICtrlToolbar_LoadBitmap ' +
        'GUICtrlToolbar_LoadImages GUICtrlToolbar_MapAccelerator ' +
        'GUICtrlToolbar_MoveButton GUICtrlToolbar_PressButton ' +
        'GUICtrlToolbar_SetAnchorHighlight GUICtrlToolbar_SetBitmapSize ' +
        'GUICtrlToolbar_SetButtonBitMap GUICtrlToolbar_SetButtonInfo ' +
        'GUICtrlToolbar_SetButtonInfoEx GUICtrlToolbar_SetButtonParam ' +
        'GUICtrlToolbar_SetButtonSize GUICtrlToolbar_SetButtonState ' +
        'GUICtrlToolbar_SetButtonStyle GUICtrlToolbar_SetButtonText ' +
        'GUICtrlToolbar_SetButtonWidth GUICtrlToolbar_SetCmdID ' +
        'GUICtrlToolbar_SetColorScheme ' +
        'GUICtrlToolbar_SetDisabledImageList ' +
        'GUICtrlToolbar_SetDrawTextFlags GUICtrlToolbar_SetExtendedStyle ' +
        'GUICtrlToolbar_SetHotImageList GUICtrlToolbar_SetHotItem ' +
        'GUICtrlToolbar_SetImageList GUICtrlToolbar_SetIndent ' +
        'GUICtrlToolbar_SetIndeterminate GUICtrlToolbar_SetInsertMark ' +
        'GUICtrlToolbar_SetInsertMarkColor GUICtrlToolbar_SetMaxTextRows ' +
        'GUICtrlToolbar_SetMetrics GUICtrlToolbar_SetPadding ' +
        'GUICtrlToolbar_SetParent GUICtrlToolbar_SetRows ' +
        'GUICtrlToolbar_SetStyle GUICtrlToolbar_SetStyleAltDrag ' +
        'GUICtrlToolbar_SetStyleCustomErase GUICtrlToolbar_SetStyleFlat ' +
        'GUICtrlToolbar_SetStyleList GUICtrlToolbar_SetStyleRegisterDrop ' +
        'GUICtrlToolbar_SetStyleToolTips ' +
        'GUICtrlToolbar_SetStyleTransparent ' +
        'GUICtrlToolbar_SetStyleWrapable GUICtrlToolbar_SetToolTips ' +
        'GUICtrlToolbar_SetUnicodeFormat GUICtrlToolbar_SetWindowTheme ' +
        'GUICtrlTreeView_Add GUICtrlTreeView_AddChild ' +
        'GUICtrlTreeView_AddChildFirst GUICtrlTreeView_AddFirst ' +
        'GUICtrlTreeView_BeginUpdate GUICtrlTreeView_ClickItem ' +
        'GUICtrlTreeView_Create GUICtrlTreeView_CreateDragImage ' +
        'GUICtrlTreeView_CreateSolidBitMap GUICtrlTreeView_Delete ' +
        'GUICtrlTreeView_DeleteAll GUICtrlTreeView_DeleteChildren ' +
        'GUICtrlTreeView_Destroy GUICtrlTreeView_DisplayRect ' +
        'GUICtrlTreeView_DisplayRectEx GUICtrlTreeView_EditText ' +
        'GUICtrlTreeView_EndEdit GUICtrlTreeView_EndUpdate ' +
        'GUICtrlTreeView_EnsureVisible GUICtrlTreeView_Expand ' +
        'GUICtrlTreeView_ExpandedOnce GUICtrlTreeView_FindItem ' +
        'GUICtrlTreeView_FindItemEx GUICtrlTreeView_GetBkColor ' +
        'GUICtrlTreeView_GetBold GUICtrlTreeView_GetChecked ' +
        'GUICtrlTreeView_GetChildCount GUICtrlTreeView_GetChildren ' +
        'GUICtrlTreeView_GetCount GUICtrlTreeView_GetCut ' +
        'GUICtrlTreeView_GetDropTarget GUICtrlTreeView_GetEditControl ' +
        'GUICtrlTreeView_GetExpanded GUICtrlTreeView_GetFirstChild ' +
        'GUICtrlTreeView_GetFirstItem GUICtrlTreeView_GetFirstVisible ' +
        'GUICtrlTreeView_GetFocused GUICtrlTreeView_GetHeight ' +
        'GUICtrlTreeView_GetImageIndex ' +
        'GUICtrlTreeView_GetImageListIconHandle ' +
        'GUICtrlTreeView_GetIndent GUICtrlTreeView_GetInsertMarkColor ' +
        'GUICtrlTreeView_GetISearchString GUICtrlTreeView_GetItemByIndex ' +
        'GUICtrlTreeView_GetItemHandle GUICtrlTreeView_GetItemParam ' +
        'GUICtrlTreeView_GetLastChild GUICtrlTreeView_GetLineColor ' +
        'GUICtrlTreeView_GetNext GUICtrlTreeView_GetNextChild ' +
        'GUICtrlTreeView_GetNextSibling GUICtrlTreeView_GetNextVisible ' +
        'GUICtrlTreeView_GetNormalImageList ' +
        'GUICtrlTreeView_GetParentHandle GUICtrlTreeView_GetParentParam ' +
        'GUICtrlTreeView_GetPrev GUICtrlTreeView_GetPrevChild ' +
        'GUICtrlTreeView_GetPrevSibling GUICtrlTreeView_GetPrevVisible ' +
        'GUICtrlTreeView_GetScrollTime GUICtrlTreeView_GetSelected ' +
        'GUICtrlTreeView_GetSelectedImageIndex ' +
        'GUICtrlTreeView_GetSelection GUICtrlTreeView_GetSiblingCount ' +
        'GUICtrlTreeView_GetState GUICtrlTreeView_GetStateImageIndex ' +
        'GUICtrlTreeView_GetStateImageList GUICtrlTreeView_GetText ' +
        'GUICtrlTreeView_GetTextColor GUICtrlTreeView_GetToolTips ' +
        'GUICtrlTreeView_GetTree GUICtrlTreeView_GetUnicodeFormat ' +
        'GUICtrlTreeView_GetVisible GUICtrlTreeView_GetVisibleCount ' +
        'GUICtrlTreeView_HitTest GUICtrlTreeView_HitTestEx ' +
        'GUICtrlTreeView_HitTestItem GUICtrlTreeView_Index ' +
        'GUICtrlTreeView_InsertItem GUICtrlTreeView_IsFirstItem ' +
        'GUICtrlTreeView_IsParent GUICtrlTreeView_Level ' +
        'GUICtrlTreeView_SelectItem GUICtrlTreeView_SelectItemByIndex ' +
        'GUICtrlTreeView_SetBkColor GUICtrlTreeView_SetBold ' +
        'GUICtrlTreeView_SetChecked GUICtrlTreeView_SetCheckedByIndex ' +
        'GUICtrlTreeView_SetChildren GUICtrlTreeView_SetCut ' +
        'GUICtrlTreeView_SetDropTarget GUICtrlTreeView_SetFocused ' +
        'GUICtrlTreeView_SetHeight GUICtrlTreeView_SetIcon ' +
        'GUICtrlTreeView_SetImageIndex GUICtrlTreeView_SetIndent ' +
        'GUICtrlTreeView_SetInsertMark ' +
        'GUICtrlTreeView_SetInsertMarkColor ' +
        'GUICtrlTreeView_SetItemHeight GUICtrlTreeView_SetItemParam ' +
        'GUICtrlTreeView_SetLineColor GUICtrlTreeView_SetNormalImageList ' +
        'GUICtrlTreeView_SetScrollTime GUICtrlTreeView_SetSelected ' +
        'GUICtrlTreeView_SetSelectedImageIndex GUICtrlTreeView_SetState ' +
        'GUICtrlTreeView_SetStateImageIndex ' +
        'GUICtrlTreeView_SetStateImageList GUICtrlTreeView_SetText ' +
        'GUICtrlTreeView_SetTextColor GUICtrlTreeView_SetToolTips ' +
        'GUICtrlTreeView_SetUnicodeFormat GUICtrlTreeView_Sort ' +
        'GUIImageList_Add GUIImageList_AddBitmap GUIImageList_AddIcon ' +
        'GUIImageList_AddMasked GUIImageList_BeginDrag ' +
        'GUIImageList_Copy GUIImageList_Create GUIImageList_Destroy ' +
        'GUIImageList_DestroyIcon GUIImageList_DragEnter ' +
        'GUIImageList_DragLeave GUIImageList_DragMove ' +
        'GUIImageList_Draw GUIImageList_DrawEx GUIImageList_Duplicate ' +
        'GUIImageList_EndDrag GUIImageList_GetBkColor ' +
        'GUIImageList_GetIcon GUIImageList_GetIconHeight ' +
        'GUIImageList_GetIconSize GUIImageList_GetIconSizeEx ' +
        'GUIImageList_GetIconWidth GUIImageList_GetImageCount ' +
        'GUIImageList_GetImageInfoEx GUIImageList_Remove ' +
        'GUIImageList_ReplaceIcon GUIImageList_SetBkColor ' +
        'GUIImageList_SetIconSize GUIImageList_SetImageCount ' +
        'GUIImageList_Swap GUIScrollBars_EnableScrollBar ' +
        'GUIScrollBars_GetScrollBarInfoEx GUIScrollBars_GetScrollBarRect ' +
        'GUIScrollBars_GetScrollBarRGState ' +
        'GUIScrollBars_GetScrollBarXYLineButton ' +
        'GUIScrollBars_GetScrollBarXYThumbBottom ' +
        'GUIScrollBars_GetScrollBarXYThumbTop ' +
        'GUIScrollBars_GetScrollInfo GUIScrollBars_GetScrollInfoEx ' +
        'GUIScrollBars_GetScrollInfoMax GUIScrollBars_GetScrollInfoMin ' +
        'GUIScrollBars_GetScrollInfoPage GUIScrollBars_GetScrollInfoPos ' +
        'GUIScrollBars_GetScrollInfoTrackPos GUIScrollBars_GetScrollPos ' +
        'GUIScrollBars_GetScrollRange GUIScrollBars_Init ' +
        'GUIScrollBars_ScrollWindow GUIScrollBars_SetScrollInfo ' +
        'GUIScrollBars_SetScrollInfoMax GUIScrollBars_SetScrollInfoMin ' +
        'GUIScrollBars_SetScrollInfoPage GUIScrollBars_SetScrollInfoPos ' +
        'GUIScrollBars_SetScrollRange GUIScrollBars_ShowScrollBar ' +
        'GUIToolTip_Activate GUIToolTip_AddTool GUIToolTip_AdjustRect ' +
        'GUIToolTip_BitsToTTF GUIToolTip_Create GUIToolTip_Deactivate ' +
        'GUIToolTip_DelTool GUIToolTip_Destroy GUIToolTip_EnumTools ' +
        'GUIToolTip_GetBubbleHeight GUIToolTip_GetBubbleSize ' +
        'GUIToolTip_GetBubbleWidth GUIToolTip_GetCurrentTool ' +
        'GUIToolTip_GetDelayTime GUIToolTip_GetMargin ' +
        'GUIToolTip_GetMarginEx GUIToolTip_GetMaxTipWidth ' +
        'GUIToolTip_GetText GUIToolTip_GetTipBkColor ' +
        'GUIToolTip_GetTipTextColor GUIToolTip_GetTitleBitMap ' +
        'GUIToolTip_GetTitleText GUIToolTip_GetToolCount ' +
        'GUIToolTip_GetToolInfo GUIToolTip_HitTest ' +
        'GUIToolTip_NewToolRect GUIToolTip_Pop GUIToolTip_PopUp ' +
        'GUIToolTip_SetDelayTime GUIToolTip_SetMargin ' +
        'GUIToolTip_SetMaxTipWidth GUIToolTip_SetTipBkColor ' +
        'GUIToolTip_SetTipTextColor GUIToolTip_SetTitle ' +
        'GUIToolTip_SetToolInfo GUIToolTip_SetWindowTheme ' +
        'GUIToolTip_ToolExists GUIToolTip_ToolToArray ' +
        'GUIToolTip_TrackActivate GUIToolTip_TrackPosition ' +
        'GUIToolTip_Update GUIToolTip_UpdateTipText HexToString ' +
        'IEAction IEAttach IEBodyReadHTML IEBodyReadText ' +
        'IEBodyWriteHTML IECreate IECreateEmbedded IEDocGetObj ' +
        'IEDocInsertHTML IEDocInsertText IEDocReadHTML ' +
        'IEDocWriteHTML IEErrorNotify IEFormElementCheckBoxSelect ' +
        'IEFormElementGetCollection IEFormElementGetObjByName ' +
        'IEFormElementGetValue IEFormElementOptionSelect ' +
        'IEFormElementRadioSelect IEFormElementSetValue ' +
        'IEFormGetCollection IEFormGetObjByName IEFormImageClick ' +
        'IEFormReset IEFormSubmit IEFrameGetCollection ' +
        'IEFrameGetObjByName IEGetObjById IEGetObjByName ' +
        'IEHeadInsertEventScript IEImgClick IEImgGetCollection ' +
        'IEIsFrameSet IELinkClickByIndex IELinkClickByText ' +
        'IELinkGetCollection IELoadWait IELoadWaitTimeout IENavigate ' +
        'IEPropertyGet IEPropertySet IEQuit IETableGetCollection ' +
        'IETableWriteToArray IETagNameAllGetCollection ' +
        'IETagNameGetCollection IE_Example IE_Introduction ' +
        'IE_VersionInfo INetExplorerCapable INetGetSource INetMail ' +
        'INetSmtpMail IsPressed MathCheckDiv Max MemGlobalAlloc ' +
        'MemGlobalFree MemGlobalLock MemGlobalSize MemGlobalUnlock ' +
        'MemMoveMemory MemVirtualAlloc MemVirtualAllocEx ' +
        'MemVirtualFree MemVirtualFreeEx Min MouseTrap ' +
        'NamedPipes_CallNamedPipe NamedPipes_ConnectNamedPipe ' +
        'NamedPipes_CreateNamedPipe NamedPipes_CreatePipe ' +
        'NamedPipes_DisconnectNamedPipe ' +
        'NamedPipes_GetNamedPipeHandleState NamedPipes_GetNamedPipeInfo ' +
        'NamedPipes_PeekNamedPipe NamedPipes_SetNamedPipeHandleState ' +
        'NamedPipes_TransactNamedPipe NamedPipes_WaitNamedPipe ' +
        'Net_Share_ConnectionEnum Net_Share_FileClose ' +
        'Net_Share_FileEnum Net_Share_FileGetInfo Net_Share_PermStr ' +
        'Net_Share_ResourceStr Net_Share_SessionDel ' +
        'Net_Share_SessionEnum Net_Share_SessionGetInfo ' +
        'Net_Share_ShareAdd Net_Share_ShareCheck Net_Share_ShareDel ' +
        'Net_Share_ShareEnum Net_Share_ShareGetInfo ' +
        'Net_Share_ShareSetInfo Net_Share_StatisticsGetSvr ' +
        'Net_Share_StatisticsGetWrk Now NowCalc NowCalcDate ' +
        'NowDate NowTime PathFull PathGetRelative PathMake ' +
        'PathSplit ProcessGetName ProcessGetPriority Radian ' +
        'ReplaceStringInFile RunDos ScreenCapture_Capture ' +
        'ScreenCapture_CaptureWnd ScreenCapture_SaveImage ' +
        'ScreenCapture_SetBMPFormat ScreenCapture_SetJPGQuality ' +
        'ScreenCapture_SetTIFColorDepth ScreenCapture_SetTIFCompression ' +
        'Security__AdjustTokenPrivileges ' +
        'Security__CreateProcessWithToken Security__DuplicateTokenEx ' +
        'Security__GetAccountSid Security__GetLengthSid ' +
        'Security__GetTokenInformation Security__ImpersonateSelf ' +
        'Security__IsValidSid Security__LookupAccountName ' +
        'Security__LookupAccountSid Security__LookupPrivilegeValue ' +
        'Security__OpenProcessToken Security__OpenThreadToken ' +
        'Security__OpenThreadTokenEx Security__SetPrivilege ' +
        'Security__SetTokenInformation Security__SidToStringSid ' +
        'Security__SidTypeStr Security__StringSidToSid SendMessage ' +
        'SendMessageA SetDate SetTime Singleton SoundClose ' +
        'SoundLength SoundOpen SoundPause SoundPlay SoundPos ' +
        'SoundResume SoundSeek SoundStatus SoundStop ' +
        'SQLite_Changes SQLite_Close SQLite_Display2DResult ' +
        'SQLite_Encode SQLite_ErrCode SQLite_ErrMsg SQLite_Escape ' +
        'SQLite_Exec SQLite_FastEncode SQLite_FastEscape ' +
        'SQLite_FetchData SQLite_FetchNames SQLite_GetTable ' +
        'SQLite_GetTable2d SQLite_LastInsertRowID SQLite_LibVersion ' +
        'SQLite_Open SQLite_Query SQLite_QueryFinalize ' +
        'SQLite_QueryReset SQLite_QuerySingleRow SQLite_SafeMode ' +
        'SQLite_SetTimeout SQLite_Shutdown SQLite_SQLiteExe ' +
        'SQLite_Startup SQLite_TotalChanges StringBetween ' +
        'StringExplode StringInsert StringProper StringRepeat ' +
        'StringTitleCase StringToHex TCPIpToName TempFile ' +
        'TicksToTime Timer_Diff Timer_GetIdleTime Timer_GetTimerID ' +
        'Timer_Init Timer_KillAllTimers Timer_KillTimer ' +
        'Timer_SetTimer TimeToTicks VersionCompare viClose ' +
        'viExecCommand viFindGpib viGpibBusReset viGTL ' +
        'viInteractiveControl viOpen viSetAttribute viSetTimeout ' +
        'WeekNumberISO WinAPI_AbortPath WinAPI_ActivateKeyboardLayout ' +
        'WinAPI_AddClipboardFormatListener WinAPI_AddFontMemResourceEx ' +
        'WinAPI_AddFontResourceEx WinAPI_AddIconOverlay ' +
        'WinAPI_AddIconTransparency WinAPI_AddMRUString ' +
        'WinAPI_AdjustBitmap WinAPI_AdjustTokenPrivileges ' +
        'WinAPI_AdjustWindowRectEx WinAPI_AlphaBlend WinAPI_AngleArc ' +
        'WinAPI_AnimateWindow WinAPI_Arc WinAPI_ArcTo ' +
        'WinAPI_ArrayToStruct WinAPI_AssignProcessToJobObject ' +
        'WinAPI_AssocGetPerceivedType WinAPI_AssocQueryString ' +
        'WinAPI_AttachConsole WinAPI_AttachThreadInput ' +
        'WinAPI_BackupRead WinAPI_BackupReadAbort WinAPI_BackupSeek ' +
        'WinAPI_BackupWrite WinAPI_BackupWriteAbort WinAPI_Beep ' +
        'WinAPI_BeginBufferedPaint WinAPI_BeginDeferWindowPos ' +
        'WinAPI_BeginPaint WinAPI_BeginPath WinAPI_BeginUpdateResource ' +
        'WinAPI_BitBlt WinAPI_BringWindowToTop ' +
        'WinAPI_BroadcastSystemMessage WinAPI_BrowseForFolderDlg ' +
        'WinAPI_BufferedPaintClear WinAPI_BufferedPaintInit ' +
        'WinAPI_BufferedPaintSetAlpha WinAPI_BufferedPaintUnInit ' +
        'WinAPI_CallNextHookEx WinAPI_CallWindowProc ' +
        'WinAPI_CallWindowProcW WinAPI_CascadeWindows ' +
        'WinAPI_ChangeWindowMessageFilterEx WinAPI_CharToOem ' +
        'WinAPI_ChildWindowFromPointEx WinAPI_ClientToScreen ' +
        'WinAPI_ClipCursor WinAPI_CloseDesktop WinAPI_CloseEnhMetaFile ' +
        'WinAPI_CloseFigure WinAPI_CloseHandle WinAPI_CloseThemeData ' +
        'WinAPI_CloseWindow WinAPI_CloseWindowStation ' +
        'WinAPI_CLSIDFromProgID WinAPI_CoInitialize ' +
        'WinAPI_ColorAdjustLuma WinAPI_ColorHLSToRGB ' +
        'WinAPI_ColorRGBToHLS WinAPI_CombineRgn ' +
        'WinAPI_CombineTransform WinAPI_CommandLineToArgv ' +
        'WinAPI_CommDlgExtendedError WinAPI_CommDlgExtendedErrorEx ' +
        'WinAPI_CompareString WinAPI_CompressBitmapBits ' +
        'WinAPI_CompressBuffer WinAPI_ComputeCrc32 ' +
        'WinAPI_ConfirmCredentials WinAPI_CopyBitmap WinAPI_CopyCursor ' +
        'WinAPI_CopyEnhMetaFile WinAPI_CopyFileEx WinAPI_CopyIcon ' +
        'WinAPI_CopyImage WinAPI_CopyRect WinAPI_CopyStruct ' +
        'WinAPI_CoTaskMemAlloc WinAPI_CoTaskMemFree ' +
        'WinAPI_CoTaskMemRealloc WinAPI_CoUninitialize ' +
        'WinAPI_Create32BitHBITMAP WinAPI_Create32BitHICON ' +
        'WinAPI_CreateANDBitmap WinAPI_CreateBitmap ' +
        'WinAPI_CreateBitmapIndirect WinAPI_CreateBrushIndirect ' +
        'WinAPI_CreateBuffer WinAPI_CreateBufferFromStruct ' +
        'WinAPI_CreateCaret WinAPI_CreateColorAdjustment ' +
        'WinAPI_CreateCompatibleBitmap WinAPI_CreateCompatibleBitmapEx ' +
        'WinAPI_CreateCompatibleDC WinAPI_CreateDesktop ' +
        'WinAPI_CreateDIB WinAPI_CreateDIBColorTable ' +
        'WinAPI_CreateDIBitmap WinAPI_CreateDIBSection ' +
        'WinAPI_CreateDirectory WinAPI_CreateDirectoryEx ' +
        'WinAPI_CreateEllipticRgn WinAPI_CreateEmptyIcon ' +
        'WinAPI_CreateEnhMetaFile WinAPI_CreateEvent WinAPI_CreateFile ' +
        'WinAPI_CreateFileEx WinAPI_CreateFileMapping ' +
        'WinAPI_CreateFont WinAPI_CreateFontEx ' +
        'WinAPI_CreateFontIndirect WinAPI_CreateGUID ' +
        'WinAPI_CreateHardLink WinAPI_CreateIcon ' +
        'WinAPI_CreateIconFromResourceEx WinAPI_CreateIconIndirect ' +
        'WinAPI_CreateJobObject WinAPI_CreateMargins ' +
        'WinAPI_CreateMRUList WinAPI_CreateMutex WinAPI_CreateNullRgn ' +
        'WinAPI_CreateNumberFormatInfo WinAPI_CreateObjectID ' +
        'WinAPI_CreatePen WinAPI_CreatePoint WinAPI_CreatePolygonRgn ' +
        'WinAPI_CreateProcess WinAPI_CreateProcessWithToken ' +
        'WinAPI_CreateRect WinAPI_CreateRectEx WinAPI_CreateRectRgn ' +
        'WinAPI_CreateRectRgnIndirect WinAPI_CreateRoundRectRgn ' +
        'WinAPI_CreateSemaphore WinAPI_CreateSize ' +
        'WinAPI_CreateSolidBitmap WinAPI_CreateSolidBrush ' +
        'WinAPI_CreateStreamOnHGlobal WinAPI_CreateString ' +
        'WinAPI_CreateSymbolicLink WinAPI_CreateTransform ' +
        'WinAPI_CreateWindowEx WinAPI_CreateWindowStation ' +
        'WinAPI_DecompressBuffer WinAPI_DecryptFile ' +
        'WinAPI_DeferWindowPos WinAPI_DefineDosDevice ' +
        'WinAPI_DefRawInputProc WinAPI_DefSubclassProc ' +
        'WinAPI_DefWindowProc WinAPI_DefWindowProcW WinAPI_DeleteDC ' +
        'WinAPI_DeleteEnhMetaFile WinAPI_DeleteFile ' +
        'WinAPI_DeleteObject WinAPI_DeleteObjectID ' +
        'WinAPI_DeleteVolumeMountPoint WinAPI_DeregisterShellHookWindow ' +
        'WinAPI_DestroyCaret WinAPI_DestroyCursor WinAPI_DestroyIcon ' +
        'WinAPI_DestroyWindow WinAPI_DeviceIoControl ' +
        'WinAPI_DisplayStruct WinAPI_DllGetVersion WinAPI_DllInstall ' +
        'WinAPI_DllUninstall WinAPI_DPtoLP WinAPI_DragAcceptFiles ' +
        'WinAPI_DragFinish WinAPI_DragQueryFileEx ' +
        'WinAPI_DragQueryPoint WinAPI_DrawAnimatedRects ' +
        'WinAPI_DrawBitmap WinAPI_DrawEdge WinAPI_DrawFocusRect ' +
        'WinAPI_DrawFrameControl WinAPI_DrawIcon WinAPI_DrawIconEx ' +
        'WinAPI_DrawLine WinAPI_DrawShadowText WinAPI_DrawText ' +
        'WinAPI_DrawThemeBackground WinAPI_DrawThemeEdge ' +
        'WinAPI_DrawThemeIcon WinAPI_DrawThemeParentBackground ' +
        'WinAPI_DrawThemeText WinAPI_DrawThemeTextEx ' +
        'WinAPI_DuplicateEncryptionInfoFile WinAPI_DuplicateHandle ' +
        'WinAPI_DuplicateTokenEx WinAPI_DwmDefWindowProc ' +
        'WinAPI_DwmEnableBlurBehindWindow WinAPI_DwmEnableComposition ' +
        'WinAPI_DwmExtendFrameIntoClientArea ' +
        'WinAPI_DwmGetColorizationColor ' +
        'WinAPI_DwmGetColorizationParameters ' +
        'WinAPI_DwmGetWindowAttribute WinAPI_DwmInvalidateIconicBitmaps ' +
        'WinAPI_DwmIsCompositionEnabled ' +
        'WinAPI_DwmQueryThumbnailSourceSize WinAPI_DwmRegisterThumbnail ' +
        'WinAPI_DwmSetColorizationParameters ' +
        'WinAPI_DwmSetIconicLivePreviewBitmap ' +
        'WinAPI_DwmSetIconicThumbnail WinAPI_DwmSetWindowAttribute ' +
        'WinAPI_DwmUnregisterThumbnail ' +
        'WinAPI_DwmUpdateThumbnailProperties WinAPI_DWordToFloat ' +
        'WinAPI_DWordToInt WinAPI_EjectMedia WinAPI_Ellipse ' +
        'WinAPI_EmptyWorkingSet WinAPI_EnableWindow WinAPI_EncryptFile ' +
        'WinAPI_EncryptionDisable WinAPI_EndBufferedPaint ' +
        'WinAPI_EndDeferWindowPos WinAPI_EndPaint WinAPI_EndPath ' +
        'WinAPI_EndUpdateResource WinAPI_EnumChildProcess ' +
        'WinAPI_EnumChildWindows WinAPI_EnumDesktops ' +
        'WinAPI_EnumDesktopWindows WinAPI_EnumDeviceDrivers ' +
        'WinAPI_EnumDisplayDevices WinAPI_EnumDisplayMonitors ' +
        'WinAPI_EnumDisplaySettings WinAPI_EnumDllProc ' +
        'WinAPI_EnumFiles WinAPI_EnumFileStreams ' +
        'WinAPI_EnumFontFamilies WinAPI_EnumHardLinks ' +
        'WinAPI_EnumMRUList WinAPI_EnumPageFiles ' +
        'WinAPI_EnumProcessHandles WinAPI_EnumProcessModules ' +
        'WinAPI_EnumProcessThreads WinAPI_EnumProcessWindows ' +
        'WinAPI_EnumRawInputDevices WinAPI_EnumResourceLanguages ' +
        'WinAPI_EnumResourceNames WinAPI_EnumResourceTypes ' +
        'WinAPI_EnumSystemGeoID WinAPI_EnumSystemLocales ' +
        'WinAPI_EnumUILanguages WinAPI_EnumWindows ' +
        'WinAPI_EnumWindowsPopup WinAPI_EnumWindowStations ' +
        'WinAPI_EnumWindowsTop WinAPI_EqualMemory WinAPI_EqualRect ' +
        'WinAPI_EqualRgn WinAPI_ExcludeClipRect ' +
        'WinAPI_ExpandEnvironmentStrings WinAPI_ExtCreatePen ' +
        'WinAPI_ExtCreateRegion WinAPI_ExtFloodFill WinAPI_ExtractIcon ' +
        'WinAPI_ExtractIconEx WinAPI_ExtSelectClipRgn ' +
        'WinAPI_FatalAppExit WinAPI_FatalExit ' +
        'WinAPI_FileEncryptionStatus WinAPI_FileExists ' +
        'WinAPI_FileIconInit WinAPI_FileInUse WinAPI_FillMemory ' +
        'WinAPI_FillPath WinAPI_FillRect WinAPI_FillRgn ' +
        'WinAPI_FindClose WinAPI_FindCloseChangeNotification ' +
        'WinAPI_FindExecutable WinAPI_FindFirstChangeNotification ' +
        'WinAPI_FindFirstFile WinAPI_FindFirstFileName ' +
        'WinAPI_FindFirstStream WinAPI_FindNextChangeNotification ' +
        'WinAPI_FindNextFile WinAPI_FindNextFileName ' +
        'WinAPI_FindNextStream WinAPI_FindResource ' +
        'WinAPI_FindResourceEx WinAPI_FindTextDlg WinAPI_FindWindow ' +
        'WinAPI_FlashWindow WinAPI_FlashWindowEx WinAPI_FlattenPath ' +
        'WinAPI_FloatToDWord WinAPI_FloatToInt WinAPI_FlushFileBuffers ' +
        'WinAPI_FlushFRBuffer WinAPI_FlushViewOfFile ' +
        'WinAPI_FormatDriveDlg WinAPI_FormatMessage WinAPI_FrameRect ' +
        'WinAPI_FrameRgn WinAPI_FreeLibrary WinAPI_FreeMemory ' +
        'WinAPI_FreeMRUList WinAPI_FreeResource WinAPI_GdiComment ' +
        'WinAPI_GetActiveWindow WinAPI_GetAllUsersProfileDirectory ' +
        'WinAPI_GetAncestor WinAPI_GetApplicationRestartSettings ' +
        'WinAPI_GetArcDirection WinAPI_GetAsyncKeyState ' +
        'WinAPI_GetBinaryType WinAPI_GetBitmapBits ' +
        'WinAPI_GetBitmapDimension WinAPI_GetBitmapDimensionEx ' +
        'WinAPI_GetBkColor WinAPI_GetBkMode WinAPI_GetBoundsRect ' +
        'WinAPI_GetBrushOrg WinAPI_GetBufferedPaintBits ' +
        'WinAPI_GetBufferedPaintDC WinAPI_GetBufferedPaintTargetDC ' +
        'WinAPI_GetBufferedPaintTargetRect WinAPI_GetBValue ' +
        'WinAPI_GetCaretBlinkTime WinAPI_GetCaretPos WinAPI_GetCDType ' +
        'WinAPI_GetClassInfoEx WinAPI_GetClassLongEx ' +
        'WinAPI_GetClassName WinAPI_GetClientHeight ' +
        'WinAPI_GetClientRect WinAPI_GetClientWidth ' +
        'WinAPI_GetClipboardSequenceNumber WinAPI_GetClipBox ' +
        'WinAPI_GetClipCursor WinAPI_GetClipRgn ' +
        'WinAPI_GetColorAdjustment WinAPI_GetCompressedFileSize ' +
        'WinAPI_GetCompression WinAPI_GetConnectedDlg ' +
        'WinAPI_GetCurrentDirectory WinAPI_GetCurrentHwProfile ' +
        'WinAPI_GetCurrentObject WinAPI_GetCurrentPosition ' +
        'WinAPI_GetCurrentProcess ' +
        'WinAPI_GetCurrentProcessExplicitAppUserModelID ' +
        'WinAPI_GetCurrentProcessID WinAPI_GetCurrentThemeName ' +
        'WinAPI_GetCurrentThread WinAPI_GetCurrentThreadId ' +
        'WinAPI_GetCursor WinAPI_GetCursorInfo WinAPI_GetDateFormat ' +
        'WinAPI_GetDC WinAPI_GetDCEx WinAPI_GetDefaultPrinter ' +
        'WinAPI_GetDefaultUserProfileDirectory WinAPI_GetDesktopWindow ' +
        'WinAPI_GetDeviceCaps WinAPI_GetDeviceDriverBaseName ' +
        'WinAPI_GetDeviceDriverFileName WinAPI_GetDeviceGammaRamp ' +
        'WinAPI_GetDIBColorTable WinAPI_GetDIBits ' +
        'WinAPI_GetDiskFreeSpaceEx WinAPI_GetDlgCtrlID ' +
        'WinAPI_GetDlgItem WinAPI_GetDllDirectory ' +
        'WinAPI_GetDriveBusType WinAPI_GetDriveGeometryEx ' +
        'WinAPI_GetDriveNumber WinAPI_GetDriveType ' +
        'WinAPI_GetDurationFormat WinAPI_GetEffectiveClientRect ' +
        'WinAPI_GetEnhMetaFile WinAPI_GetEnhMetaFileBits ' +
        'WinAPI_GetEnhMetaFileDescription WinAPI_GetEnhMetaFileDimension ' +
        'WinAPI_GetEnhMetaFileHeader WinAPI_GetErrorMessage ' +
        'WinAPI_GetErrorMode WinAPI_GetExitCodeProcess ' +
        'WinAPI_GetExtended WinAPI_GetFileAttributes WinAPI_GetFileID ' +
        'WinAPI_GetFileInformationByHandle ' +
        'WinAPI_GetFileInformationByHandleEx WinAPI_GetFilePointerEx ' +
        'WinAPI_GetFileSizeEx WinAPI_GetFileSizeOnDisk ' +
        'WinAPI_GetFileTitle WinAPI_GetFileType ' +
        'WinAPI_GetFileVersionInfo WinAPI_GetFinalPathNameByHandle ' +
        'WinAPI_GetFinalPathNameByHandleEx WinAPI_GetFocus ' +
        'WinAPI_GetFontMemoryResourceInfo WinAPI_GetFontName ' +
        'WinAPI_GetFontResourceInfo WinAPI_GetForegroundWindow ' +
        'WinAPI_GetFRBuffer WinAPI_GetFullPathName WinAPI_GetGeoInfo ' +
        'WinAPI_GetGlyphOutline WinAPI_GetGraphicsMode ' +
        'WinAPI_GetGuiResources WinAPI_GetGUIThreadInfo ' +
        'WinAPI_GetGValue WinAPI_GetHandleInformation ' +
        'WinAPI_GetHGlobalFromStream WinAPI_GetIconDimension ' +
        'WinAPI_GetIconInfo WinAPI_GetIconInfoEx WinAPI_GetIdleTime ' +
        'WinAPI_GetKeyboardLayout WinAPI_GetKeyboardLayoutList ' +
        'WinAPI_GetKeyboardState WinAPI_GetKeyboardType ' +
        'WinAPI_GetKeyNameText WinAPI_GetKeyState ' +
        'WinAPI_GetLastActivePopup WinAPI_GetLastError ' +
        'WinAPI_GetLastErrorMessage WinAPI_GetLayeredWindowAttributes ' +
        'WinAPI_GetLocaleInfo WinAPI_GetLogicalDrives ' +
        'WinAPI_GetMapMode WinAPI_GetMemorySize ' +
        'WinAPI_GetMessageExtraInfo WinAPI_GetModuleFileNameEx ' +
        'WinAPI_GetModuleHandle WinAPI_GetModuleHandleEx ' +
        'WinAPI_GetModuleInformation WinAPI_GetMonitorInfo ' +
        'WinAPI_GetMousePos WinAPI_GetMousePosX WinAPI_GetMousePosY ' +
        'WinAPI_GetMUILanguage WinAPI_GetNumberFormat WinAPI_GetObject ' +
        'WinAPI_GetObjectID WinAPI_GetObjectInfoByHandle ' +
        'WinAPI_GetObjectNameByHandle WinAPI_GetObjectType ' +
        'WinAPI_GetOpenFileName WinAPI_GetOutlineTextMetrics ' +
        'WinAPI_GetOverlappedResult WinAPI_GetParent ' +
        'WinAPI_GetParentProcess WinAPI_GetPerformanceInfo ' +
        'WinAPI_GetPEType WinAPI_GetPhysicallyInstalledSystemMemory ' +
        'WinAPI_GetPixel WinAPI_GetPolyFillMode WinAPI_GetPosFromRect ' +
        'WinAPI_GetPriorityClass WinAPI_GetProcAddress ' +
        'WinAPI_GetProcessAffinityMask WinAPI_GetProcessCommandLine ' +
        'WinAPI_GetProcessFileName WinAPI_GetProcessHandleCount ' +
        'WinAPI_GetProcessID WinAPI_GetProcessIoCounters ' +
        'WinAPI_GetProcessMemoryInfo WinAPI_GetProcessName ' +
        'WinAPI_GetProcessShutdownParameters WinAPI_GetProcessTimes ' +
        'WinAPI_GetProcessUser WinAPI_GetProcessWindowStation ' +
        'WinAPI_GetProcessWorkingDirectory WinAPI_GetProfilesDirectory ' +
        'WinAPI_GetPwrCapabilities WinAPI_GetRawInputBuffer ' +
        'WinAPI_GetRawInputBufferLength WinAPI_GetRawInputData ' +
        'WinAPI_GetRawInputDeviceInfo WinAPI_GetRegionData ' +
        'WinAPI_GetRegisteredRawInputDevices ' +
        'WinAPI_GetRegKeyNameByHandle WinAPI_GetRgnBox WinAPI_GetROP2 ' +
        'WinAPI_GetRValue WinAPI_GetSaveFileName WinAPI_GetShellWindow ' +
        'WinAPI_GetStartupInfo WinAPI_GetStdHandle ' +
        'WinAPI_GetStockObject WinAPI_GetStretchBltMode ' +
        'WinAPI_GetString WinAPI_GetSysColor WinAPI_GetSysColorBrush ' +
        'WinAPI_GetSystemDefaultLangID WinAPI_GetSystemDefaultLCID ' +
        'WinAPI_GetSystemDefaultUILanguage WinAPI_GetSystemDEPPolicy ' +
        'WinAPI_GetSystemInfo WinAPI_GetSystemMetrics ' +
        'WinAPI_GetSystemPowerStatus WinAPI_GetSystemTimes ' +
        'WinAPI_GetSystemWow64Directory WinAPI_GetTabbedTextExtent ' +
        'WinAPI_GetTempFileName WinAPI_GetTextAlign ' +
        'WinAPI_GetTextCharacterExtra WinAPI_GetTextColor ' +
        'WinAPI_GetTextExtentPoint32 WinAPI_GetTextFace ' +
        'WinAPI_GetTextMetrics WinAPI_GetThemeAppProperties ' +
        'WinAPI_GetThemeBackgroundContentRect ' +
        'WinAPI_GetThemeBackgroundExtent WinAPI_GetThemeBackgroundRegion ' +
        'WinAPI_GetThemeBitmap WinAPI_GetThemeBool ' +
        'WinAPI_GetThemeColor WinAPI_GetThemeDocumentationProperty ' +
        'WinAPI_GetThemeEnumValue WinAPI_GetThemeFilename ' +
        'WinAPI_GetThemeFont WinAPI_GetThemeInt WinAPI_GetThemeMargins ' +
        'WinAPI_GetThemeMetric WinAPI_GetThemePartSize ' +
        'WinAPI_GetThemePosition WinAPI_GetThemePropertyOrigin ' +
        'WinAPI_GetThemeRect WinAPI_GetThemeString ' +
        'WinAPI_GetThemeSysBool WinAPI_GetThemeSysColor ' +
        'WinAPI_GetThemeSysColorBrush WinAPI_GetThemeSysFont ' +
        'WinAPI_GetThemeSysInt WinAPI_GetThemeSysSize ' +
        'WinAPI_GetThemeSysString WinAPI_GetThemeTextExtent ' +
        'WinAPI_GetThemeTextMetrics WinAPI_GetThemeTransitionDuration ' +
        'WinAPI_GetThreadDesktop WinAPI_GetThreadErrorMode ' +
        'WinAPI_GetThreadLocale WinAPI_GetThreadUILanguage ' +
        'WinAPI_GetTickCount WinAPI_GetTickCount64 ' +
        'WinAPI_GetTimeFormat WinAPI_GetTopWindow ' +
        'WinAPI_GetUDFColorMode WinAPI_GetUpdateRect ' +
        'WinAPI_GetUpdateRgn WinAPI_GetUserDefaultLangID ' +
        'WinAPI_GetUserDefaultLCID WinAPI_GetUserDefaultUILanguage ' +
        'WinAPI_GetUserGeoID WinAPI_GetUserObjectInformation ' +
        'WinAPI_GetVersion WinAPI_GetVersionEx ' +
        'WinAPI_GetVolumeInformation WinAPI_GetVolumeInformationByHandle ' +
        'WinAPI_GetVolumeNameForVolumeMountPoint WinAPI_GetWindow ' +
        'WinAPI_GetWindowDC WinAPI_GetWindowDisplayAffinity ' +
        'WinAPI_GetWindowExt WinAPI_GetWindowFileName ' +
        'WinAPI_GetWindowHeight WinAPI_GetWindowInfo ' +
        'WinAPI_GetWindowLong WinAPI_GetWindowOrg ' +
        'WinAPI_GetWindowPlacement WinAPI_GetWindowRect ' +
        'WinAPI_GetWindowRgn WinAPI_GetWindowRgnBox ' +
        'WinAPI_GetWindowSubclass WinAPI_GetWindowText ' +
        'WinAPI_GetWindowTheme WinAPI_GetWindowThreadProcessId ' +
        'WinAPI_GetWindowWidth WinAPI_GetWorkArea ' +
        'WinAPI_GetWorldTransform WinAPI_GetXYFromPoint ' +
        'WinAPI_GlobalMemoryStatus WinAPI_GradientFill ' +
        'WinAPI_GUIDFromString WinAPI_GUIDFromStringEx WinAPI_HashData ' +
        'WinAPI_HashString WinAPI_HiByte WinAPI_HideCaret ' +
        'WinAPI_HiDWord WinAPI_HiWord WinAPI_InflateRect ' +
        'WinAPI_InitMUILanguage WinAPI_InProcess ' +
        'WinAPI_IntersectClipRect WinAPI_IntersectRect ' +
        'WinAPI_IntToDWord WinAPI_IntToFloat WinAPI_InvalidateRect ' +
        'WinAPI_InvalidateRgn WinAPI_InvertANDBitmap ' +
        'WinAPI_InvertColor WinAPI_InvertRect WinAPI_InvertRgn ' +
        'WinAPI_IOCTL WinAPI_IsAlphaBitmap WinAPI_IsBadCodePtr ' +
        'WinAPI_IsBadReadPtr WinAPI_IsBadStringPtr ' +
        'WinAPI_IsBadWritePtr WinAPI_IsChild WinAPI_IsClassName ' +
        'WinAPI_IsDoorOpen WinAPI_IsElevated WinAPI_IsHungAppWindow ' +
        'WinAPI_IsIconic WinAPI_IsInternetConnected ' +
        'WinAPI_IsLoadKBLayout WinAPI_IsMemory ' +
        'WinAPI_IsNameInExpression WinAPI_IsNetworkAlive ' +
        'WinAPI_IsPathShared WinAPI_IsProcessInJob ' +
        'WinAPI_IsProcessorFeaturePresent WinAPI_IsRectEmpty ' +
        'WinAPI_IsThemeActive ' +
        'WinAPI_IsThemeBackgroundPartiallyTransparent ' +
        'WinAPI_IsThemePartDefined WinAPI_IsValidLocale ' +
        'WinAPI_IsWindow WinAPI_IsWindowEnabled WinAPI_IsWindowUnicode ' +
        'WinAPI_IsWindowVisible WinAPI_IsWow64Process ' +
        'WinAPI_IsWritable WinAPI_IsZoomed WinAPI_Keybd_Event ' +
        'WinAPI_KillTimer WinAPI_LineDDA WinAPI_LineTo ' +
        'WinAPI_LoadBitmap WinAPI_LoadCursor WinAPI_LoadCursorFromFile ' +
        'WinAPI_LoadIcon WinAPI_LoadIconMetric ' +
        'WinAPI_LoadIconWithScaleDown WinAPI_LoadImage ' +
        'WinAPI_LoadIndirectString WinAPI_LoadKeyboardLayout ' +
        'WinAPI_LoadLibrary WinAPI_LoadLibraryEx WinAPI_LoadMedia ' +
        'WinAPI_LoadResource WinAPI_LoadShell32Icon WinAPI_LoadString ' +
        'WinAPI_LoadStringEx WinAPI_LoByte WinAPI_LocalFree ' +
        'WinAPI_LockDevice WinAPI_LockFile WinAPI_LockResource ' +
        'WinAPI_LockWindowUpdate WinAPI_LockWorkStation WinAPI_LoDWord ' +
        'WinAPI_LongMid WinAPI_LookupIconIdFromDirectoryEx ' +
        'WinAPI_LoWord WinAPI_LPtoDP WinAPI_MAKELANGID ' +
        'WinAPI_MAKELCID WinAPI_MakeLong WinAPI_MakeQWord ' +
        'WinAPI_MakeWord WinAPI_MapViewOfFile WinAPI_MapVirtualKey ' +
        'WinAPI_MaskBlt WinAPI_MessageBeep WinAPI_MessageBoxCheck ' +
        'WinAPI_MessageBoxIndirect WinAPI_MirrorIcon ' +
        'WinAPI_ModifyWorldTransform WinAPI_MonitorFromPoint ' +
        'WinAPI_MonitorFromRect WinAPI_MonitorFromWindow ' +
        'WinAPI_Mouse_Event WinAPI_MoveFileEx WinAPI_MoveMemory ' +
        'WinAPI_MoveTo WinAPI_MoveToEx WinAPI_MoveWindow ' +
        'WinAPI_MsgBox WinAPI_MulDiv WinAPI_MultiByteToWideChar ' +
        'WinAPI_MultiByteToWideCharEx WinAPI_NtStatusToDosError ' +
        'WinAPI_OemToChar WinAPI_OffsetClipRgn WinAPI_OffsetPoints ' +
        'WinAPI_OffsetRect WinAPI_OffsetRgn WinAPI_OffsetWindowOrg ' +
        'WinAPI_OpenDesktop WinAPI_OpenFileById WinAPI_OpenFileDlg ' +
        'WinAPI_OpenFileMapping WinAPI_OpenIcon ' +
        'WinAPI_OpenInputDesktop WinAPI_OpenJobObject WinAPI_OpenMutex ' +
        'WinAPI_OpenProcess WinAPI_OpenProcessToken ' +
        'WinAPI_OpenSemaphore WinAPI_OpenThemeData ' +
        'WinAPI_OpenWindowStation WinAPI_PageSetupDlg ' +
        'WinAPI_PaintDesktop WinAPI_PaintRgn WinAPI_ParseURL ' +
        'WinAPI_ParseUserName WinAPI_PatBlt WinAPI_PathAddBackslash ' +
        'WinAPI_PathAddExtension WinAPI_PathAppend ' +
        'WinAPI_PathBuildRoot WinAPI_PathCanonicalize ' +
        'WinAPI_PathCommonPrefix WinAPI_PathCompactPath ' +
        'WinAPI_PathCompactPathEx WinAPI_PathCreateFromUrl ' +
        'WinAPI_PathFindExtension WinAPI_PathFindFileName ' +
        'WinAPI_PathFindNextComponent WinAPI_PathFindOnPath ' +
        'WinAPI_PathGetArgs WinAPI_PathGetCharType ' +
        'WinAPI_PathGetDriveNumber WinAPI_PathIsContentType ' +
        'WinAPI_PathIsDirectory WinAPI_PathIsDirectoryEmpty ' +
        'WinAPI_PathIsExe WinAPI_PathIsFileSpec ' +
        'WinAPI_PathIsLFNFileSpec WinAPI_PathIsRelative ' +
        'WinAPI_PathIsRoot WinAPI_PathIsSameRoot ' +
        'WinAPI_PathIsSystemFolder WinAPI_PathIsUNC ' +
        'WinAPI_PathIsUNCServer WinAPI_PathIsUNCServerShare ' +
        'WinAPI_PathMakeSystemFolder WinAPI_PathMatchSpec ' +
        'WinAPI_PathParseIconLocation WinAPI_PathRelativePathTo ' +
        'WinAPI_PathRemoveArgs WinAPI_PathRemoveBackslash ' +
        'WinAPI_PathRemoveExtension WinAPI_PathRemoveFileSpec ' +
        'WinAPI_PathRenameExtension WinAPI_PathSearchAndQualify ' +
        'WinAPI_PathSkipRoot WinAPI_PathStripPath ' +
        'WinAPI_PathStripToRoot WinAPI_PathToRegion ' +
        'WinAPI_PathUndecorate WinAPI_PathUnExpandEnvStrings ' +
        'WinAPI_PathUnmakeSystemFolder WinAPI_PathUnquoteSpaces ' +
        'WinAPI_PathYetAnotherMakeUniqueName WinAPI_PickIconDlg ' +
        'WinAPI_PlayEnhMetaFile WinAPI_PlaySound WinAPI_PlgBlt ' +
        'WinAPI_PointFromRect WinAPI_PolyBezier WinAPI_PolyBezierTo ' +
        'WinAPI_PolyDraw WinAPI_Polygon WinAPI_PostMessage ' +
        'WinAPI_PrimaryLangId WinAPI_PrintDlg WinAPI_PrintDlgEx ' +
        'WinAPI_PrintWindow WinAPI_ProgIDFromCLSID WinAPI_PtInRect ' +
        'WinAPI_PtInRectEx WinAPI_PtInRegion WinAPI_PtVisible ' +
        'WinAPI_QueryDosDevice WinAPI_QueryInformationJobObject ' +
        'WinAPI_QueryPerformanceCounter WinAPI_QueryPerformanceFrequency ' +
        'WinAPI_RadialGradientFill WinAPI_ReadDirectoryChanges ' +
        'WinAPI_ReadFile WinAPI_ReadProcessMemory WinAPI_Rectangle ' +
        'WinAPI_RectInRegion WinAPI_RectIsEmpty WinAPI_RectVisible ' +
        'WinAPI_RedrawWindow WinAPI_RegCloseKey ' +
        'WinAPI_RegConnectRegistry WinAPI_RegCopyTree ' +
        'WinAPI_RegCopyTreeEx WinAPI_RegCreateKey ' +
        'WinAPI_RegDeleteEmptyKey WinAPI_RegDeleteKey ' +
        'WinAPI_RegDeleteKeyValue WinAPI_RegDeleteTree ' +
        'WinAPI_RegDeleteTreeEx WinAPI_RegDeleteValue ' +
        'WinAPI_RegDisableReflectionKey WinAPI_RegDuplicateHKey ' +
        'WinAPI_RegEnableReflectionKey WinAPI_RegEnumKey ' +
        'WinAPI_RegEnumValue WinAPI_RegFlushKey ' +
        'WinAPI_RegisterApplicationRestart WinAPI_RegisterClass ' +
        'WinAPI_RegisterClassEx WinAPI_RegisterHotKey ' +
        'WinAPI_RegisterPowerSettingNotification ' +
        'WinAPI_RegisterRawInputDevices WinAPI_RegisterShellHookWindow ' +
        'WinAPI_RegisterWindowMessage WinAPI_RegLoadMUIString ' +
        'WinAPI_RegNotifyChangeKeyValue WinAPI_RegOpenKey ' +
        'WinAPI_RegQueryInfoKey WinAPI_RegQueryLastWriteTime ' +
        'WinAPI_RegQueryMultipleValues WinAPI_RegQueryReflectionKey ' +
        'WinAPI_RegQueryValue WinAPI_RegRestoreKey WinAPI_RegSaveKey ' +
        'WinAPI_RegSetValue WinAPI_ReleaseCapture WinAPI_ReleaseDC ' +
        'WinAPI_ReleaseMutex WinAPI_ReleaseSemaphore ' +
        'WinAPI_ReleaseStream WinAPI_RemoveClipboardFormatListener ' +
        'WinAPI_RemoveDirectory WinAPI_RemoveFontMemResourceEx ' +
        'WinAPI_RemoveFontResourceEx WinAPI_RemoveWindowSubclass ' +
        'WinAPI_ReOpenFile WinAPI_ReplaceFile WinAPI_ReplaceTextDlg ' +
        'WinAPI_ResetEvent WinAPI_RestartDlg WinAPI_RestoreDC ' +
        'WinAPI_RGB WinAPI_RotatePoints WinAPI_RoundRect ' +
        'WinAPI_SaveDC WinAPI_SaveFileDlg WinAPI_SaveHBITMAPToFile ' +
        'WinAPI_SaveHICONToFile WinAPI_ScaleWindowExt ' +
        'WinAPI_ScreenToClient WinAPI_SearchPath WinAPI_SelectClipPath ' +
        'WinAPI_SelectClipRgn WinAPI_SelectObject ' +
        'WinAPI_SendMessageTimeout WinAPI_SetActiveWindow ' +
        'WinAPI_SetArcDirection WinAPI_SetBitmapBits ' +
        'WinAPI_SetBitmapDimensionEx WinAPI_SetBkColor ' +
        'WinAPI_SetBkMode WinAPI_SetBoundsRect WinAPI_SetBrushOrg ' +
        'WinAPI_SetCapture WinAPI_SetCaretBlinkTime WinAPI_SetCaretPos ' +
        'WinAPI_SetClassLongEx WinAPI_SetColorAdjustment ' +
        'WinAPI_SetCompression WinAPI_SetCurrentDirectory ' +
        'WinAPI_SetCurrentProcessExplicitAppUserModelID WinAPI_SetCursor ' +
        'WinAPI_SetDCBrushColor WinAPI_SetDCPenColor ' +
        'WinAPI_SetDefaultPrinter WinAPI_SetDeviceGammaRamp ' +
        'WinAPI_SetDIBColorTable WinAPI_SetDIBits ' +
        'WinAPI_SetDIBitsToDevice WinAPI_SetDllDirectory ' +
        'WinAPI_SetEndOfFile WinAPI_SetEnhMetaFileBits ' +
        'WinAPI_SetErrorMode WinAPI_SetEvent WinAPI_SetFileAttributes ' +
        'WinAPI_SetFileInformationByHandleEx WinAPI_SetFilePointer ' +
        'WinAPI_SetFilePointerEx WinAPI_SetFileShortName ' +
        'WinAPI_SetFileValidData WinAPI_SetFocus WinAPI_SetFont ' +
        'WinAPI_SetForegroundWindow WinAPI_SetFRBuffer ' +
        'WinAPI_SetGraphicsMode WinAPI_SetHandleInformation ' +
        'WinAPI_SetInformationJobObject WinAPI_SetKeyboardLayout ' +
        'WinAPI_SetKeyboardState WinAPI_SetLastError ' +
        'WinAPI_SetLayeredWindowAttributes WinAPI_SetLocaleInfo ' +
        'WinAPI_SetMapMode WinAPI_SetMessageExtraInfo WinAPI_SetParent ' +
        'WinAPI_SetPixel WinAPI_SetPolyFillMode ' +
        'WinAPI_SetPriorityClass WinAPI_SetProcessAffinityMask ' +
        'WinAPI_SetProcessShutdownParameters ' +
        'WinAPI_SetProcessWindowStation WinAPI_SetRectRgn ' +
        'WinAPI_SetROP2 WinAPI_SetSearchPathMode ' +
        'WinAPI_SetStretchBltMode WinAPI_SetSysColors ' +
        'WinAPI_SetSystemCursor WinAPI_SetTextAlign ' +
        'WinAPI_SetTextCharacterExtra WinAPI_SetTextColor ' +
        'WinAPI_SetTextJustification WinAPI_SetThemeAppProperties ' +
        'WinAPI_SetThreadDesktop WinAPI_SetThreadErrorMode ' +
        'WinAPI_SetThreadExecutionState WinAPI_SetThreadLocale ' +
        'WinAPI_SetThreadUILanguage WinAPI_SetTimer ' +
        'WinAPI_SetUDFColorMode WinAPI_SetUserGeoID ' +
        'WinAPI_SetUserObjectInformation WinAPI_SetVolumeMountPoint ' +
        'WinAPI_SetWindowDisplayAffinity WinAPI_SetWindowExt ' +
        'WinAPI_SetWindowLong WinAPI_SetWindowOrg ' +
        'WinAPI_SetWindowPlacement WinAPI_SetWindowPos ' +
        'WinAPI_SetWindowRgn WinAPI_SetWindowsHookEx ' +
        'WinAPI_SetWindowSubclass WinAPI_SetWindowText ' +
        'WinAPI_SetWindowTheme WinAPI_SetWinEventHook ' +
        'WinAPI_SetWorldTransform WinAPI_SfcIsFileProtected ' +
        'WinAPI_SfcIsKeyProtected WinAPI_ShellAboutDlg ' +
        'WinAPI_ShellAddToRecentDocs WinAPI_ShellChangeNotify ' +
        'WinAPI_ShellChangeNotifyDeregister ' +
        'WinAPI_ShellChangeNotifyRegister WinAPI_ShellCreateDirectory ' +
        'WinAPI_ShellEmptyRecycleBin WinAPI_ShellExecute ' +
        'WinAPI_ShellExecuteEx WinAPI_ShellExtractAssociatedIcon ' +
        'WinAPI_ShellExtractIcon WinAPI_ShellFileOperation ' +
        'WinAPI_ShellFlushSFCache WinAPI_ShellGetFileInfo ' +
        'WinAPI_ShellGetIconOverlayIndex WinAPI_ShellGetImageList ' +
        'WinAPI_ShellGetKnownFolderIDList WinAPI_ShellGetKnownFolderPath ' +
        'WinAPI_ShellGetLocalizedName WinAPI_ShellGetPathFromIDList ' +
        'WinAPI_ShellGetSetFolderCustomSettings WinAPI_ShellGetSettings ' +
        'WinAPI_ShellGetSpecialFolderLocation ' +
        'WinAPI_ShellGetSpecialFolderPath WinAPI_ShellGetStockIconInfo ' +
        'WinAPI_ShellILCreateFromPath WinAPI_ShellNotifyIcon ' +
        'WinAPI_ShellNotifyIconGetRect WinAPI_ShellObjectProperties ' +
        'WinAPI_ShellOpenFolderAndSelectItems WinAPI_ShellOpenWithDlg ' +
        'WinAPI_ShellQueryRecycleBin ' +
        'WinAPI_ShellQueryUserNotificationState ' +
        'WinAPI_ShellRemoveLocalizedName WinAPI_ShellRestricted ' +
        'WinAPI_ShellSetKnownFolderPath WinAPI_ShellSetLocalizedName ' +
        'WinAPI_ShellSetSettings WinAPI_ShellStartNetConnectionDlg ' +
        'WinAPI_ShellUpdateImage WinAPI_ShellUserAuthenticationDlg ' +
        'WinAPI_ShellUserAuthenticationDlgEx WinAPI_ShortToWord ' +
        'WinAPI_ShowCaret WinAPI_ShowCursor WinAPI_ShowError ' +
        'WinAPI_ShowLastError WinAPI_ShowMsg WinAPI_ShowOwnedPopups ' +
        'WinAPI_ShowWindow WinAPI_ShutdownBlockReasonCreate ' +
        'WinAPI_ShutdownBlockReasonDestroy ' +
        'WinAPI_ShutdownBlockReasonQuery WinAPI_SizeOfResource ' +
        'WinAPI_StretchBlt WinAPI_StretchDIBits ' +
        'WinAPI_StrFormatByteSize WinAPI_StrFormatByteSizeEx ' +
        'WinAPI_StrFormatKBSize WinAPI_StrFromTimeInterval ' +
        'WinAPI_StringFromGUID WinAPI_StringLenA WinAPI_StringLenW ' +
        'WinAPI_StrLen WinAPI_StrokeAndFillPath WinAPI_StrokePath ' +
        'WinAPI_StructToArray WinAPI_SubLangId WinAPI_SubtractRect ' +
        'WinAPI_SwapDWord WinAPI_SwapQWord WinAPI_SwapWord ' +
        'WinAPI_SwitchColor WinAPI_SwitchDesktop ' +
        'WinAPI_SwitchToThisWindow WinAPI_SystemParametersInfo ' +
        'WinAPI_TabbedTextOut WinAPI_TerminateJobObject ' +
        'WinAPI_TerminateProcess WinAPI_TextOut WinAPI_TileWindows ' +
        'WinAPI_TrackMouseEvent WinAPI_TransparentBlt ' +
        'WinAPI_TwipsPerPixelX WinAPI_TwipsPerPixelY ' +
        'WinAPI_UnhookWindowsHookEx WinAPI_UnhookWinEvent ' +
        'WinAPI_UnionRect WinAPI_UnionStruct WinAPI_UniqueHardwareID ' +
        'WinAPI_UnloadKeyboardLayout WinAPI_UnlockFile ' +
        'WinAPI_UnmapViewOfFile WinAPI_UnregisterApplicationRestart ' +
        'WinAPI_UnregisterClass WinAPI_UnregisterHotKey ' +
        'WinAPI_UnregisterPowerSettingNotification ' +
        'WinAPI_UpdateLayeredWindow WinAPI_UpdateLayeredWindowEx ' +
        'WinAPI_UpdateLayeredWindowIndirect WinAPI_UpdateResource ' +
        'WinAPI_UpdateWindow WinAPI_UrlApplyScheme ' +
        'WinAPI_UrlCanonicalize WinAPI_UrlCombine WinAPI_UrlCompare ' +
        'WinAPI_UrlCreateFromPath WinAPI_UrlFixup WinAPI_UrlGetPart ' +
        'WinAPI_UrlHash WinAPI_UrlIs WinAPI_UserHandleGrantAccess ' +
        'WinAPI_ValidateRect WinAPI_ValidateRgn WinAPI_VerQueryRoot ' +
        'WinAPI_VerQueryValue WinAPI_VerQueryValueEx ' +
        'WinAPI_WaitForInputIdle WinAPI_WaitForMultipleObjects ' +
        'WinAPI_WaitForSingleObject WinAPI_WideCharToMultiByte ' +
        'WinAPI_WidenPath WinAPI_WindowFromDC WinAPI_WindowFromPoint ' +
        'WinAPI_WordToShort WinAPI_Wow64EnableWow64FsRedirection ' +
        'WinAPI_WriteConsole WinAPI_WriteFile ' +
        'WinAPI_WriteProcessMemory WinAPI_ZeroMemory ' +
        'WinNet_AddConnection WinNet_AddConnection2 ' +
        'WinNet_AddConnection3 WinNet_CancelConnection ' +
        'WinNet_CancelConnection2 WinNet_CloseEnum ' +
        'WinNet_ConnectionDialog WinNet_ConnectionDialog1 ' +
        'WinNet_DisconnectDialog WinNet_DisconnectDialog1 ' +
        'WinNet_EnumResource WinNet_GetConnection ' +
        'WinNet_GetConnectionPerformance WinNet_GetLastError ' +
        'WinNet_GetNetworkInformation WinNet_GetProviderName ' +
        'WinNet_GetResourceInformation WinNet_GetResourceParent ' +
        'WinNet_GetUniversalName WinNet_GetUser WinNet_OpenEnum ' +
        'WinNet_RestoreConnection WinNet_UseConnection Word_Create ' +
        'Word_DocAdd Word_DocAttach Word_DocClose Word_DocExport ' +
        'Word_DocFind Word_DocFindReplace Word_DocGet ' +
        'Word_DocLinkAdd Word_DocLinkGet Word_DocOpen ' +
        'Word_DocPictureAdd Word_DocPrint Word_DocRangeSet ' +
        'Word_DocSave Word_DocSaveAs Word_DocTableRead ' +
        'Word_DocTableWrite Word_Quit',

        COMMENT = {
            variants: [
              hljs.COMMENT(';', '$', {relevance: 0}),
              hljs.COMMENT('#cs', '#ce'),
              hljs.COMMENT('#comments-start', '#comments-end')
            ]
        },

        VARIABLE = {
            className: 'variable',
            begin: '\\$[A-z0-9_]+'
        },

        STRING = {
            className: 'string',
            variants: [{
                begin: /"/,
                end: /"/,
                contains: [{
                    begin: /""/,
                    relevance: 0
                }]
            }, {
                begin: /'/,
                end: /'/,
                contains: [{
                    begin: /''/,
                    relevance: 0
                }]
            }]
        },

        NUMBER = {
            variants: [hljs.BINARY_NUMBER_MODE, hljs.C_NUMBER_MODE]
        },

        PREPROCESSOR = {
            className: 'preprocessor',
            begin: '#',
            end: '$',
            keywords: 'include include-once NoTrayIcon OnAutoItStartRegister RequireAdmin pragma ' +
                'Au3Stripper_Ignore_Funcs Au3Stripper_Ignore_Variables ' +
                'Au3Stripper_Off Au3Stripper_On Au3Stripper_Parameters ' +
                'AutoIt3Wrapper_Add_Constants AutoIt3Wrapper_Au3Check_Parameters ' +
                'AutoIt3Wrapper_Au3Check_Stop_OnWarning AutoIt3Wrapper_Aut2Exe ' +
                'AutoIt3Wrapper_AutoIt3 AutoIt3Wrapper_AutoIt3Dir ' +
                'AutoIt3Wrapper_Change2CUI AutoIt3Wrapper_Compile_Both ' +
                'AutoIt3Wrapper_Compression AutoIt3Wrapper_EndIf ' +
                'AutoIt3Wrapper_Icon AutoIt3Wrapper_If_Compile ' +
                'AutoIt3Wrapper_If_Run AutoIt3Wrapper_Jump_To_First_Error ' +
                'AutoIt3Wrapper_OutFile AutoIt3Wrapper_OutFile_Type ' +
                'AutoIt3Wrapper_OutFile_X64 AutoIt3Wrapper_PlugIn_Funcs ' +
                'AutoIt3Wrapper_Res_Comment Autoit3Wrapper_Res_Compatibility ' +
                'AutoIt3Wrapper_Res_Description AutoIt3Wrapper_Res_Field ' +
                'AutoIt3Wrapper_Res_File_Add AutoIt3Wrapper_Res_FileVersion ' +
                'AutoIt3Wrapper_Res_FileVersion_AutoIncrement ' +
                'AutoIt3Wrapper_Res_Icon_Add AutoIt3Wrapper_Res_Language ' +
                'AutoIt3Wrapper_Res_LegalCopyright ' +
                'AutoIt3Wrapper_Res_ProductVersion ' +
                'AutoIt3Wrapper_Res_requestedExecutionLevel ' +
                'AutoIt3Wrapper_Res_SaveSource AutoIt3Wrapper_Run_After ' +
                'AutoIt3Wrapper_Run_Au3Check AutoIt3Wrapper_Run_Au3Stripper ' +
                'AutoIt3Wrapper_Run_Before AutoIt3Wrapper_Run_Debug_Mode ' +
                'AutoIt3Wrapper_Run_SciTE_Minimized ' +
                'AutoIt3Wrapper_Run_SciTE_OutputPane_Minimized ' +
                'AutoIt3Wrapper_Run_Tidy AutoIt3Wrapper_ShowProgress ' +
                'AutoIt3Wrapper_Testing AutoIt3Wrapper_Tidy_Stop_OnError ' +
                'AutoIt3Wrapper_UPX_Parameters AutoIt3Wrapper_UseUPX ' +
                'AutoIt3Wrapper_UseX64 AutoIt3Wrapper_Version ' +
                'AutoIt3Wrapper_Versioning AutoIt3Wrapper_Versioning_Parameters ' +
                'Tidy_Off Tidy_On Tidy_Parameters EndRegion Region',
            contains: [{
                    begin: /\\\n/,
                    relevance: 0
                }, {
                    beginKeywords: 'include',
                    end: '$',
                    contains: [
                        STRING, {
                            className: 'string',
                            variants: [{
                                begin: '<',
                                end: '>'
                            }, {
                                begin: /"/,
                                end: /"/,
                                contains: [{
                                    begin: /""/,
                                    relevance: 0
                                }]
                            }, {
                                begin: /'/,
                                end: /'/,
                                contains: [{
                                    begin: /''/,
                                    relevance: 0
                                }]
                            }]
                        }
                    ]
                },
                STRING,
                COMMENT
            ]
        },

        CONSTANT = {
            className: 'constant',
            // begin: '@',
            // end: '$',
            // keywords: 'AppDataCommonDir AppDataDir AutoItExe AutoItPID AutoItVersion AutoItX64 COM_EventObj CommonFilesDir Compiled ComputerName ComSpec CPUArch CR CRLF DesktopCommonDir DesktopDepth DesktopDir DesktopHeight DesktopRefresh DesktopWidth DocumentsCommonDir error exitCode exitMethod extended FavoritesCommonDir FavoritesDir GUI_CtrlHandle GUI_CtrlId GUI_DragFile GUI_DragId GUI_DropId GUI_WinHandle HomeDrive HomePath HomeShare HotKeyPressed HOUR IPAddress1 IPAddress2 IPAddress3 IPAddress4 KBLayout LF LocalAppDataDir LogonDNSDomain LogonDomain LogonServer MDAY MIN MON MSEC MUILang MyDocumentsDir NumParams OSArch OSBuild OSLang OSServicePack OSType OSVersion ProgramFilesDir ProgramsCommonDir ProgramsDir ScriptDir ScriptFullPath ScriptLineNumber ScriptName SEC StartMenuCommonDir StartMenuDir StartupCommonDir StartupDir SW_DISABLE SW_ENABLE SW_HIDE SW_LOCK SW_MAXIMIZE SW_MINIMIZE SW_RESTORE SW_SHOW SW_SHOWDEFAULT SW_SHOWMAXIMIZED SW_SHOWMINIMIZED SW_SHOWMINNOACTIVE SW_SHOWNA SW_SHOWNOACTIVATE SW_SHOWNORMAL SW_UNLOCK SystemDir TAB TempDir TRAY_ID TrayIconFlashing TrayIconVisible UserName UserProfileDir WDAY WindowsDir WorkingDir YDAY YEAR',
            // relevance: 5
            begin: '@[A-z0-9_]+'
        },

        FUNCTION = {
            className: 'function',
            beginKeywords: 'Func',
            end: '$',
            excludeEnd: true,
            illegal: '\\$|\\[|%',
            contains: [
                hljs.UNDERSCORE_TITLE_MODE, {
                    className: 'params',
                    begin: '\\(',
                    end: '\\)',
                    contains: [
                        VARIABLE,
                        STRING,
                        NUMBER
                    ]
                }
            ]
        };

    return {
        case_insensitive: true,
        keywords: {
            keyword: KEYWORDS,
            built_in: BUILT_IN,
            literal: LITERAL
        },
        contains: [
            COMMENT,
            VARIABLE,
            STRING,
            NUMBER,
            PREPROCESSOR,
            CONSTANT,
            FUNCTION
        ]
    }
}
