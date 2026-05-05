import json
import psutil
import os
import gc
import tracemalloc
from datetime import datetime
import numpy as np

class MemoryProfiler:
    """  Memory profiler for Python scripts with detailed tracking"""
    
    def __init__(self, name="experiment", track_objects=True, snapshot_interval=100):
        """
        name: experiment name
        track_objects: track Python objects (via tracemalloc)
        snapshot_interval: interval for object snapshots (in iterations)
        """
        self.name = name
        self.track_objects = track_objects
        self.snapshot_interval = snapshot_interval
        self.metrics = []
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.gc_stats = []
        
        # For tracemalloc
        if track_objects:
            tracemalloc.start()
            # Take initial snapshot after initialization to avoid capturing profiler setup overhead
            self.initial_snapshot = tracemalloc.take_snapshot()
    
    def snapshot(self, label="", iteration=0, extra_info=None):
        """Capture current memory state readings"""
        current_time = datetime.now()
        
        memory_info = self.process.memory_info()
        # Cache full memory info to avoid multiple calls
        memory_full = self.process.memory_full_info() if hasattr(self.process, 'memory_full_info') else None
        cpu_times = self.process.cpu_times()
        
        # Collect metrics
        snapshot_data = {
            'timestamp': current_time.isoformat(),
            'timestamp_sec': current_time.timestamp(),
            'iteration': iteration,
            'label': label,
            
            # Process Memory
            'memory_rss_mb': memory_info.rss / 1024 / 1024,
            'memory_vms_mb': memory_info.vms / 1024 / 1024,
            'memory_uss_mb': memory_full.uss / 1024 / 1024 if memory_full else None,
            'memory_pss_mb': memory_full.pss / 1024 / 1024 if memory_full else None,
            'memory_shared_mb': memory_full.shared / 1024 / 1024 if memory_full else None,
            
            # Swap
            'swap_mb': memory_full.swap / 1024 / 1024 if memory_full else None,
            
            # CPU
            'cpu_percent': self.process.cpu_percent(interval=None),
            'cpu_user_sec': cpu_times.user,
            'cpu_system_sec': cpu_times.system,
            
            # Threads and File Descriptors
            'num_threads': self.process.num_threads(),
            'num_fds': self.process.num_fds() if hasattr(self.process, 'num_fds') else None,
            
            'system_memory_percent': self.process.memory_percent(),
            'system_memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
            
            'extra_info': extra_info
        }
        
        # GC Statistics collection (Removed forced gc.collect() to prevent distorting memory growth)
        # Forced GC often hides memory leaks by cleaning up temporary objects prematurely.
        if iteration % 100 == 0:  # Reduced frequency to minimize impact
            gc_stats = gc.get_stats()
            self.gc_stats.append({
                'iteration': iteration,
                'timestamp': current_time.isoformat(),
                'collections': gc_stats
            })
            snapshot_data['gc_collections'] = len(gc_stats)
        
        self.metrics.append(snapshot_data)
        
        # Tracking Python objects (heavy operation)
        if self.track_objects and iteration % self.snapshot_interval == 0:
            current_snapshot = tracemalloc.take_snapshot()
            # Compare to initial snapshot to see growth/leaks
            top_stats = current_snapshot.compare_to(self.initial_snapshot, 'lineno')
            snapshot_data['top_objects'] = [
                {'size_kb': stat.size_diff / 1024, 
                 'count': stat.count_diff,
                 'traceback': str(stat.traceback)[:200]} 
                for stat in top_stats[:10] if stat.size_diff != 0
            ]
            # Stop the current snapshot to free memory
            del current_snapshot
        
        return snapshot_data
    
    def get_memory_stats(self):
        """Get memory statistics"""
        if not self.metrics:
            return {}
        
        rss_values = [m['memory_rss_mb'] for m in self.metrics]
        
        return {
            'memory_peak_mb': max(rss_values),
            'memory_min_mb': min(rss_values),
            'memory_avg_mb': np.mean(rss_values),
            'memory_std_mb': np.std(rss_values),
            'memory_final_mb': rss_values[-1],
            'memory_start_mb': rss_values[0],
            'memory_increase_mb': rss_values[-1] - rss_values[0],
            'total_iterations': len(self.metrics)
        }
    
    def save(self, filename=None):
        """Save metrics to JSON"""
        if filename is None:
            # Ensure logs directory exists
            os.makedirs('logs', exist_ok=True)
            filename = f'logs/metrics_{self.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        output = {
            'experiment_name': self.name,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': datetime.now().isoformat(),
            'metrics': self.metrics,
            'gc_stats': self.gc_stats,
            'summary_stats': self.get_memory_stats()
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"Metrics saved to {filename}")
        return filename
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        # Stop tracemalloc to free resources
        if self.track_objects:
            tracemalloc.stop()
        self.save()


def create_beautiful_plot(metrics_file, output_image='memory_analysis.png'):
    """Create a beautiful plot from saved metrics"""
    import warnings

    warnings.filterwarnings('ignore')
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    # Load data
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data['metrics'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['elapsed_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    
    stats = data['summary_stats']
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. RSS Memory Plot (Main)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['elapsed_seconds'], df['memory_rss_mb'], 
             linewidth=2.5, color='#2E86AB', label='RSS (Physical Memory)')
    ax1.fill_between(df['elapsed_seconds'], df['memory_rss_mb'], alpha=0.3, color='#2E86AB')
    
    # Add VMS if available
    if df['memory_vms_mb'].notna().all():
        ax1.plot(df['elapsed_seconds'], df['memory_vms_mb'], 
                linewidth=1.5, color='#A23B72', alpha=0.7, linestyle='--', label='VMS (Virtual Memory)')
    
    # Add USS if available
    if df['memory_uss_mb'].notna().all():
        ax1.plot(df['elapsed_seconds'], df['memory_uss_mb'], 
                linewidth=1.5, color='#F18F01', alpha=0.7, linestyle=':', label='USS (Unique Memory)')
    
    # Mark memory peak
    peak_idx = df['memory_rss_mb'].idxmax()
    peak_time = df.loc[peak_idx, 'elapsed_seconds']
    peak_value = df.loc[peak_idx, 'memory_rss_mb']
    ax1.scatter(peak_time, peak_value, color='red', s=100, zorder=5, 
               edgecolors='white', linewidth=2, label=f'Peak: {peak_value:.1f} MB')
    ax1.axhline(y=peak_value, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    # Annotation with peak
    ax1.annotate(f'Memory Peak: {peak_value:.1f} MB', 
                xy=(peak_time, peak_value),
                xytext=(10, 30), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                arrowprops=dict(arrowstyle="->", color='red', lw=1.5))
    
    ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Memory (MB)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Memory Profile: {data["experiment_name"]}\n'
                  f'Peak: {stats["memory_peak_mb"]:.1f} MB | '
                  f'Avg: {stats["memory_avg_mb"]:.1f} MB | '
                  f'Growth: {stats["memory_increase_mb"]:+.1f} MB',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # 2. CPU Usage
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df['elapsed_seconds'], df['cpu_percent'], 
             linewidth=2, color='#73AB84', marker='o', markersize=4, alpha=0.7)
    ax2.fill_between(df['elapsed_seconds'], df['cpu_percent'], alpha=0.3, color='#73AB84')
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('CPU (%)', fontsize=11)
    ax2.set_title('CPU Load', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    # Handle case where cpu_percent might be 0 or NaN
    max_cpu = df['cpu_percent'].max()
    ax2.set_ylim(0, max_cpu * 1.2 if max_cpu > 0 else 100)
    
    # 3. Thread Count
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df['elapsed_seconds'], df['num_threads'], 
             linewidth=2, color='#A23B72', marker='s', markersize=4, alpha=0.7)
    ax3.fill_between(df['elapsed_seconds'], df['num_threads'], alpha=0.3, color='#A23B72')
    ax3.set_xlabel('Time (seconds)', fontsize=11)
    ax3.set_ylabel('Thread Count', fontsize=11)
    ax3.set_title('Execution Threads', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    max_threads = df['num_threads'].max()
    ax3.set_ylim(0, max_threads * 1.2 if max_threads > 0 else 10)
    
    # 4. Memory Distribution Histogram
    ax4 = fig.add_subplot(gs[2, 0])
    sns.histplot(df['memory_rss_mb'], bins=20, kde=True, ax=ax4, color='#2E86AB')
    ax4.axvline(stats['memory_avg_mb'], color='red', linestyle='--', 
                linewidth=2, label=f'Avg: {stats["memory_avg_mb"]:.1f} MB')
    ax4.axvline(stats['memory_peak_mb'], color='green', linestyle='--', 
                linewidth=2, label=f'Peak: {stats["memory_peak_mb"]:.1f} MB')
    ax4.set_xlabel('Memory (MB)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Memory Usage Distribution', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Statistical Summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Create text summary
    stats_text = f"""
    MEMORY USAGE STATISTICS
    
    ┌─────────────────────────────────────────┐
    │  Peak Usage:                            │
    │    {stats['memory_peak_mb']:>8.1f} MB              │
    ├─────────────────────────────────────────┤
    │  Average Usage:                         │
    │    {stats['memory_avg_mb']:>8.1f} MB              │
    ├─────────────────────────────────────────┤
    │  Standard Deviation:                    │
    │    {stats['memory_std_mb']:>8.1f} MB              │
    ├─────────────────────────────────────────┤
    │  Start Value:                           │
    │    {stats['memory_start_mb']:>8.1f} MB              │
    ├─────────────────────────────────────────┤
    │  End Value:                             │
    │    {stats['memory_final_mb']:>8.1f} MB              │
    ├─────────────────────────────────────────┤
    │  Memory Change:                         │
    │    {stats['memory_increase_mb']:>+8.1f} MB              │
    ├─────────────────────────────────────────┤
    │  Total Measurements:                    │
    │    {stats['total_iterations']:>8d}                 │
    └─────────────────────────────────────────┘
    
    Additional Info:
    • Experiment: {data['experiment_name']}
    • Start: {data['start_time'][:19] if data['start_time'] else 'N/A'}
    • End: {data['end_time'][:19]}
    """
    
    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#CCCCCC"))
    
    fig.suptitle(f'Memory Usage Analysis: {data["experiment_name"]}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_image}")
    plt.close(fig)
    
    return fig