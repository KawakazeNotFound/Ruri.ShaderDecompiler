using System.Runtime.InteropServices;

namespace Ruri.ShaderTools.Pipeline.Batching;

/// <summary>
/// Admits CPU work while the machine has headroom. When the runtime reports high
/// memory load, already-running jobs drain and new jobs run one at a time.
///
/// Concurrency is ADMISSION-GATED rather than fixed. A batch starts at one worker
/// and the monitor raises the ceiling each time it samples system CPU below the
/// cap. The ceiling never shrinks: a job already running cannot be un-started, so
/// the only lever is refusing to start new ones — and taking that lever away
/// again would just stall the batch.
///
/// Starting at one and climbing (rather than starting wide and hoping) is what
/// keeps a bulk decompile from monopolising a machine someone is using.
///
/// CPU sampling is Windows-specific. The runtime's memory signal is used even
/// on platforms without a CPU sampler.
/// </summary>
internal sealed class CpuAdmissionGate : IDisposable
{
    /// <summary>How often system load is resampled.</summary>
    private const int SampleIntervalMs = 750;

    private readonly object _lock = new();
    private readonly int _ceiling;
    private readonly int _cpuCapPercent;
    private readonly CancellationTokenSource _monitorCts;
    private readonly Task _monitor;

    private int _allowed;
    private int _inFlight;
    private bool _highMemoryLoad;

    public CpuAdmissionGate(int ceiling, int cpuCapPercent, CancellationToken cancellationToken)
    {
        _ceiling = Math.Max(1, ceiling);
        _cpuCapPercent = Math.Clamp(cpuCapPercent, 1, 100);
        // The cap is a share of the machine, so that share is what runs from the first job on.
        // Starting from one job and granting another per sample left a batch of three hundred
        // shaders effectively serial for most of its life: the ramp restarted with every batch
        // and reached the cores only after seconds. The monitor still grants beyond the share
        // while the machine sits idle, and never takes a running job back.
        _allowed = Math.Clamp(_ceiling * _cpuCapPercent / 100, 1, _ceiling);
        _highMemoryLoad = IsHighMemoryLoad();

        _monitorCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        _monitor = Task.Run(() => MonitorLoop(_monitorCts.Token));
    }

    /// <summary>Token that fires when the batch is torn down.</summary>
    public CancellationToken MonitorToken => _monitorCts.Token;

    /// <summary>
    /// Block until a slot is free. Returns false when the batch was cancelled
    /// while waiting.
    /// </summary>
    public bool Acquire(CancellationToken cancellationToken)
    {
        lock (_lock)
        {
            while (_inFlight >= (_highMemoryLoad ? 1 : _allowed))
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    return false;
                }
                Monitor.Wait(_lock, 250);
            }

            if (cancellationToken.IsCancellationRequested) return false;
            _inFlight++;
            return true;
        }
    }

    public void Release()
    {
        lock (_lock)
        {
            if (_inFlight > 0)
            {
                _inFlight--;
            }
            Monitor.PulseAll(_lock);
        }
    }

    private void GrantOneMore()
    {
        lock (_lock)
        {
            if (_allowed < _ceiling)
            {
                _allowed++;
                Monitor.PulseAll(_lock);
            }
        }
    }

    private void OpenFully()
    {
        lock (_lock)
        {
            _allowed = _ceiling;
            Monitor.PulseAll(_lock);
        }
    }

    private static bool IsHighMemoryLoad()
    {
        var memory = GC.GetGCMemoryInfo();
        return memory.HighMemoryLoadThresholdBytes > 0 && memory.MemoryLoadBytes >= memory.HighMemoryLoadThresholdBytes;
    }

    private void RefreshMemoryPressure()
    {
        bool highMemoryLoad = IsHighMemoryLoad();
        lock (_lock)
        {
            if (highMemoryLoad == _highMemoryLoad) return;
            _highMemoryLoad = highMemoryLoad;
            Monitor.PulseAll(_lock);
        }
    }

    private void MonitorLoop(CancellationToken cancellationToken)
    {
        // Without a baseline the first sample reads as 100% busy and the gate
        // would never open.
        bool sampleCpu = TrySampleCpuTimes(out long previousIdle, out long previousKernel, out long previousUser);
        if (!sampleCpu)
        {
            OpenFully();
        }

        // Wait on the token's handle rather than Task.Delay(...).Wait(): the
        // latter raises a first-chance cancellation exception every time a batch
        // finishes normally, which is pure debugger noise. WaitOne signals via its
        // return value — true means "token fired, stop", false means "interval
        // elapsed, continue".
        WaitHandle cancelHandle = cancellationToken.WaitHandle;

        while (!cancellationToken.IsCancellationRequested)
        {
            if (cancelHandle.WaitOne(SampleIntervalMs))
            {
                return;
            }

            RefreshMemoryPressure();
            if (!sampleCpu || !TrySampleCpuTimes(out long idle, out long kernel, out long user))
            {
                continue;
            }

            long idleDelta = idle - previousIdle;
            long totalDelta = (kernel - previousKernel) + (user - previousUser);
            previousIdle = idle;
            previousKernel = kernel;
            previousUser = user;

            if (totalDelta <= 0)
            {
                continue;
            }

            // GetSystemTimes reports kernel time INCLUSIVE of idle, so the
            // wall-clock envelope is kernel + user and idle is subtracted from it.
            double usage = 1.0 - ((double)idleDelta / totalDelta);
            if ((int)Math.Round(usage * 100) < _cpuCapPercent)
            {
                GrantOneMore();
            }

            // At or above the cap: leave the allowance alone. Running jobs finish;
            // we simply stop admitting more.
        }
    }

    private static bool TrySampleCpuTimes(out long idle, out long kernel, out long user)
    {
        idle = 0;
        kernel = 0;
        user = 0;

        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return false;
        }

        try
        {
            return GetSystemTimes(out idle, out kernel, out user);
        }
        catch
        {
            return false;
        }
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetSystemTimes(out long lpIdleTime, out long lpKernelTime, out long lpUserTime);

    public void Dispose()
    {
        _monitorCts.Cancel();
        try
        {
            _monitor.Wait();
        }
        catch
        {
            // The monitor is advisory; a fault in it must not fail the batch.
        }
        _monitorCts.Dispose();
    }
}
