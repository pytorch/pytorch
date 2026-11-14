// C# 5 / .NET Framework 4.7.2. Child output never crosses a PowerShell binding.
using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace PyTorch.Ci {
public static class WindowsIdentityLauncher {
    const string Python = @"C:\Jenkins\Miniconda3\python.exe";
    const int Limit = 4096;
    static readonly string[] Keys = ("windows_administrator ambient_credential_inputs_present metadata_reachable runner_instance_identity_matched instance_profile_matched instance_role_matched temporary_credentials_present sts_authenticated own_instance_describe_allowed own_instance_describe_denied own_instance_identity_matched own_instance_environment_matched own_instance_dynamic own_root_device_ebs own_root_delete_on_termination own_other_ebs_mapping_present own_all_ebs_delete_on_termination own_ephemeral_finished_tag_present own_root_replacement_tag_present probe_error").Split(' ');
    static readonly UTF8Encoding Utf8 = new UTF8Encoding(false, true);
    public sealed class Report {
        public readonly string Json;
        public readonly int ExitCode;
        public bool ChildStopped { get; internal set; }
        internal Report(string json, int exitCode) { Json = json; ExitCode = exitCode; }
    }
    static string Serialize(bool[] values) {
        var text = new StringBuilder("{");
        for (int i = 0; i < Keys.Length; i++) {
            if (i != 0) text.Append(',');
            text.Append('"').Append(Keys[i]).Append("\":").Append(values[i] ? "true" : "false");
        }
        return text.Append('}').ToString();
    }
    public static Report Failure() {
        var values = new bool[Keys.Length]; values[values.Length - 1] = true;
        return new Report(Serialize(values), 1);
    }
    static Report Parse(byte[] data, int exitCode) {
        if (data == null || data.Length > Limit) return Failure();
        string text = Utf8.GetString(data); int offset = 0;
        var values = new bool[Keys.Length];
        for (int i = 0; i < Keys.Length; i++) {
            string prefix = (i == 0 ? "{" : ",") + "\"" + Keys[i] + "\":";
            if (offset + prefix.Length > text.Length || string.CompareOrdinal(text, offset, prefix, 0, prefix.Length) != 0) return Failure();
            offset += prefix.Length;
            if (offset + 4 <= text.Length && string.CompareOrdinal(text, offset, "true", 0, 4) == 0) { values[i] = true; offset += 4; }
            else if (offset + 5 <= text.Length && string.CompareOrdinal(text, offset, "false", 0, 5) == 0) offset += 5;
            else return Failure();
        }
        string tail = text.Substring(offset);
        if (tail != "}\n" && tail != "}\r\n") return Failure();
        int expected = values[values.Length - 1] ? 1 : 0;
        return exitCode == expected ? new Report(Serialize(values), expected) : Failure();
    }
    static async Task<byte[]> ReadBounded(Stream stream, int limit, TaskCompletionSource<bool> invalid, CancellationToken cancel) {
        try {
            using (var output = new MemoryStream()) {
                var buffer = new byte[1024];
                while (true) {
                    int count = await stream.ReadAsync(buffer, 0, Math.Min(buffer.Length, limit + 1 - (int)output.Length), cancel).ConfigureAwait(false);
                    if (count == 0) return output.ToArray();
                    if (output.Length + count > limit) { invalid.TrySetResult(true); return null; }
                    output.Write(buffer, 0, count);
                }
            }
        } catch { invalid.TrySetResult(true); return null; }
    }
    static bool StopOwned(Process process) {
        try { if (!process.HasExited) process.Kill(); } catch { }
        try { return process.WaitForExit(5000) && process.HasExited; } catch { return false; }
    }
    static Report Collect(ProcessStartInfo start, int milliseconds) {
        Report report = Failure();
        using (var process = new Process())
        using (var cancel = new CancellationTokenSource()) {
            bool started = false;
            try {
                start.UseShellExecute = false; start.CreateNoWindow = true;
                start.RedirectStandardOutput = true; start.RedirectStandardError = true;
                process.StartInfo = start; process.EnableRaisingEvents = true;
                var exited = new TaskCompletionSource<bool>();
                var invalid = new TaskCompletionSource<bool>();
                process.Exited += delegate { exited.TrySetResult(true); };
                started = process.Start();
                if (started) {
                    if (process.HasExited) exited.TrySetResult(true);
                    Task<byte[]> stdout = ReadBounded(process.StandardOutput.BaseStream, Limit, invalid, cancel.Token);
                    Task<byte[]> stderr = ReadBounded(process.StandardError.BaseStream, Limit, invalid, cancel.Token);
                    Task complete = Task.WhenAll(exited.Task, stdout, stderr);
                    Task winner = Task.WhenAny(complete, invalid.Task, Task.Delay(milliseconds)).GetAwaiter().GetResult();
                    if (winner == complete) {
                        complete.GetAwaiter().GetResult();
                        byte[] error = stderr.GetAwaiter().GetResult();
                        if (error != null && error.Length == 0) report = Parse(stdout.GetAwaiter().GetResult(), process.ExitCode);
                    }
                }
            } catch { report = Failure(); }
            finally {
                cancel.Cancel();
                report.ChildStopped = !started || StopOwned(process);
            }
        }
        if (!report.ChildStopped) return Failure();
        return report;
    }
    public static Report Run(string directory) {
        try {
            if (!File.Exists(Python)) return Failure();
            string script = Path.GetFullPath(Path.Combine(directory, "ci_windows_identity.py"));
            if (script.IndexOf('"') >= 0) return Failure();
            return Collect(new ProcessStartInfo(Python, "-I -B -S \"" + script + "\""), 120000);
        } catch { return Failure(); }
    }
    // Synthetic process tests call the same collector. Run() has no fallback.
    public static Report CollectForTest(ProcessStartInfo start, int milliseconds) {
        if (start == null || milliseconds < 1 || milliseconds > 120000) return Failure();
        return Collect(start, milliseconds);
    }
}
}
