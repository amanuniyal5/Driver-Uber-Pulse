"""
Driver Pulse - Pipeline Runner
==============================
Runs all 4 pipelines in sequence:
1. Motion Event Detection (BoVW)
2. Audio Conflict/Stress Detection (4-Layer)
3. Signal Fusion
4. Earnings Velocity Forecast

Usage:
    python run_all_pipelines.py [--pipeline N]

Options:
    --pipeline N    Run only pipeline N (1-4)
    --all           Run all pipelines (default)
"""

import os
import sys
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    """Print the Driver Pulse banner."""
    print("\n" + "=" * 70)
    print("  ____       _                  ____        _          ")
    print(" |  _ \\ _ __(_)_   _____ _ __  |  _ \\ _   _| |___  ___ ")
    print(" | | | | '__| \\ \\ / / _ \\ '__| | |_) | | | | / __|/ _ \\")
    print(" | |_| | |  | |\\ V /  __/ |    |  __/| |_| | \\__ \\  __/")
    print(" |____/|_|  |_| \\_/ \\___|_|    |_|    \\__,_|_|___/\\___|")
    print("=" * 70)
    print("  Comprehensive Driver Safety & Earnings Analytics Platform")
    print("=" * 70 + "\n")


def run_pipeline_1():
    """Run Motion Event Detection Pipeline."""
    print("\n" + "=" * 70)
    print("🚗 PIPELINE 1: Motion Event Detection (BoVW)")
    print("=" * 70)
    
    from pipelines.pipeline1_motion_bovw import main as motion_main
    motion_main()


def run_pipeline_2():
    """Run Audio Detection Pipeline."""
    print("\n" + "=" * 70)
    print("🎤 PIPELINE 2: Audio Conflict/Stress Detection (4-Layer)")
    print("=" * 70)
    
    from pipelines.pipeline2_audio_4layer import main as audio_main
    audio_main()


def run_pipeline_3():
    """Run Signal Fusion Pipeline."""
    print("\n" + "=" * 70)
    print("🔗 PIPELINE 3: Signal Fusion")
    print("=" * 70)
    
    from pipelines.pipeline3_signal_fusion import main as fusion_main
    fusion_main()


def run_pipeline_4():
    """Run Earnings Forecast Pipeline."""
    print("\n" + "=" * 70)
    print("💰 PIPELINE 4: Earnings Velocity Forecast")
    print("=" * 70)
    
    from pipelines.pipeline4_earnings_forecast import main as earnings_main
    earnings_main()


def run_all_pipelines():
    """Run all pipelines in sequence."""
    start_time = datetime.now()
    
    print_banner()
    print(f"Starting pipeline execution at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        run_pipeline_1()
        run_pipeline_2()
        run_pipeline_3()
        run_pipeline_4()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("✅ ALL PIPELINES COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Total execution time: {duration:.1f} seconds")
        print(f"Outputs saved to: outputs/")
        print("\nGenerated files:")
        
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
        if os.path.exists(output_dir):
            for f in sorted(os.listdir(output_dir)):
                filepath = os.path.join(output_dir, f)
                size = os.path.getsize(filepath)
                print(f"  • {f} ({size:,} bytes)")
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(description='Driver Pulse Pipeline Runner')
    parser.add_argument('--pipeline', type=int, choices=[1, 2, 3, 4],
                        help='Run specific pipeline (1-4)')
    parser.add_argument('--all', action='store_true', default=True,
                        help='Run all pipelines (default)')
    
    args = parser.parse_args()
    
    if args.pipeline:
        print_banner()
        print(f"Running Pipeline {args.pipeline} only...")
        
        if args.pipeline == 1:
            run_pipeline_1()
        elif args.pipeline == 2:
            run_pipeline_2()
        elif args.pipeline == 3:
            run_pipeline_3()
        elif args.pipeline == 4:
            run_pipeline_4()
    else:
        run_all_pipelines()


if __name__ == '__main__':
    main()
