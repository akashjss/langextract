#!/usr/bin/env python3
"""
Demo script for the new file upload feature in LangExtract Gradio app.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Launch Gradio with file upload demo."""
    try:
        from gradio_app import create_gradio_interface
        
        print("🎉 LangExtract Gradio - Now with File Upload Support!")
        print("=" * 60)
        print()
        print("🆕 NEW FEATURES:")
        print("📁 PDF Upload - Automatic text extraction from PDF files")
        print("📄 Word Docs - Support for .docx documents")
        print("📝 Text Files - Direct import of .txt and .md files")
        print("📊 File Info - Size and word count display")
        print("⚡ Real-time - Instant text extraction on upload")
        print()
        print("📋 Supported Formats:")
        print("   • PDF files (.pdf)")
        print("   • Word documents (.docx)")
        print("   • Text files (.txt, .md)")
        print("   • Size limit: 10MB maximum")
        print()
        print("🚀 How to Use:")
        print("1. Launch the interface")
        print("2. Go to the 'Extraction' tab")
        print("3. Click 'Upload Document' button")
        print("4. Select your PDF, Word doc, or text file")
        print("5. Text will auto-fill in the input field")
        print("6. Add examples and extract!")
        print()
        print("🌐 Works with all models:")
        print("   • Local Ollama models (gemma2, llama, etc.)")
        print("   • Cloud models (Gemini, OpenAI)")
        print()
        print("Starting interface...")
        print("Visit http://localhost:7860 to test file uploads!")
        print("Press Ctrl+C to stop")
        print()
        
        # Create and launch interface
        interface = create_gradio_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True,
            quiet=False
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Missing dependencies. Install with:")
        print("   pip install 'langextract[gradio]'")
        return 1
        
    except KeyboardInterrupt:
        print("\n👋 Stopping demo...")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
