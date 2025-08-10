#!/usr/bin/env python3
"""
Simple launcher script for the LangExtract Gradio app.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from gradio_app import main

    if __name__ == "__main__":
        print("🔍 LangExtract Gradio Interface")
        print("===============================")
        print("Starting web interface...")
        print("Visit http://localhost:7860 once started")
        print("Press Ctrl+C to stop")
        print()

        main()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\n💡 Make sure to install Gradio:")
    print("   pip install 'langextract[gradio]'")
    print("   OR")
    print("   pip install gradio>=4.0.0")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n👋 Shutting down LangExtract Gradio interface...")
    sys.exit(0)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
