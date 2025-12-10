{ Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
file LICENSE.rst or https://cmake.org/licensing for details. }

function CPackGetCustomInstallationMessage(Param: String): String;
begin
  Result := SetupMessage(msgCustomInstallation);
end;

{ Downloaded components }
#ifdef CPackDownloadCount
const
  NO_PROGRESS_BOX = 4;
  RESPOND_YES_TO_ALL = 16;
var
  CPackDownloadPage: TDownloadWizardPage;
  CPackShell: Variant;

<event('InitializeWizard')>
procedure CPackInitializeWizard();
begin
  CPackDownloadPage := CreateDownloadPage(SetupMessage(msgWizardPreparing), SetupMessage(msgPreparingDesc), nil);
  CPackShell := CreateOleObject('Shell.Application');
end;

<event('NextButtonClick')>
function CPackNextButtonClick(CurPageID: Integer): Boolean;
begin
  if CurPageID = wpReady then
  begin
    CPackDownloadPage.Clear;
    CPackDownloadPage.Show;

#sub AddDownload
  if WizardIsComponentSelected('{#CPackDownloadComponents[i]}') then
    #emit "CPackDownloadPage.Add('" + CPackDownloadUrls[i] + "', '" + CPackDownloadArchives[i] + ".zip', '" + CPackDownloadHashes[i] + "');"
#endsub
#define i
#for {i = 0; i < CPackDownloadCount; i++} AddDownload
#undef i

    try
      try
        CPackDownloadPage.Download;
        Result := True;
      except
        if not CPackDownloadPage.AbortedByUser then
          SuppressibleMsgBox(AddPeriod(GetExceptionMessage), mbCriticalError, MB_OK, IDOK);

        Result := False;
      end;
    finally
      CPackDownloadPage.Hide;
    end;
  end else
    Result := True;
end;

procedure CPackExtractFile(ArchiveName, FileName: String);
var
  ZipFileName: String;
  ZipFile: Variant;
  Item: Variant;
  TargetFolderName: String;
  TargetFolder: Variant;
begin
  TargetFolderName := RemoveBackslashUnlessRoot(ExpandConstant('{tmp}\' + ArchiveName + '\' + ExtractFileDir(FileName)));
  ZipFileName := ExpandConstant('{tmp}\' + ArchiveName + '.zip');

  if not DirExists(TargetFolderName) then
    if not ForceDirectories(TargetFolderName) then
      RaiseException(Format('Target path "%s" cannot be created', [TargetFolderName]));

  ZipFile := CPackShell.NameSpace(ZipFileName);
  if VarIsClear(ZipFile) then
    RaiseException(Format('Cannot open ZIP file "%s" or does not exist', [ZipFileName]));

  Item := ZipFile.ParseName(FileName);
  if VarIsClear(Item) then
    RaiseException(Format('Cannot find "%s" in "%s" ZIP file', [FileName, ZipFileName]));

  TargetFolder := CPackShell.NameSpace(TargetFolderName);
  if VarIsClear(TargetFolder) then
    RaiseException(Format('Target path "%s" does not exist', [TargetFolderName]));

  TargetFolder.CopyHere(Item, NO_PROGRESS_BOX or RESPOND_YES_TO_ALL);
end;

#endif
