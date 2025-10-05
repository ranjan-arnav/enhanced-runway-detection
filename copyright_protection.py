#!/usr/bin/env python3
"""
🚨 COPYRIGHT PROTECTION SCRIPT 🚨

This script displays copyright warnings when anyone attempts to run the code.

Copyright (c) 2025 Arnav Ranjan. All Rights Reserved.
PROPRIETARY AND CONFIDENTIAL - UNAUTHORIZED USE PROHIBITED
"""

import sys
import time

def display_copyright_warning():
    """Display comprehensive copyright warning"""
    
    warning_message = """
    
🚨🚨🚨 COPYRIGHT PROTECTION WARNING 🚨🚨🚨

╔══════════════════════════════════════════════════════════════╗
║                    ⚠️  LEGAL NOTICE  ⚠️                     ║
║                                                              ║
║  This software is PROPRIETARY and CONFIDENTIAL             ║
║  Copyright (c) 2025 Arnav Ranjan. All Rights Reserved.     ║
║                                                              ║
║  🚫 UNAUTHORIZED USE IS STRICTLY PROHIBITED 🚫              ║
║                                                              ║
║  By running this code, you acknowledge:                     ║
║  • You are authorized to view this code only               ║
║  • You will not modify, copy, or distribute this code     ║
║  • You will not use this code for commercial purposes     ║
║  • You understand violations may result in legal action   ║
║                                                              ║
║  📧 Contact: Arnav Ranjan for licensing inquiries          ║
╚══════════════════════════════════════════════════════════════╝

⏰ This warning will display for 10 seconds...

    """
    
    print(warning_message)
    
    # Countdown timer
    for i in range(10, 0, -1):
        print(f"⏳ Continuing in {i} seconds... (Press Ctrl+C to stop)")
        time.sleep(1)
    
    print("\n✅ Proceeding with authorized viewing...")
    return True

def check_authorization():
    """Check if user acknowledges copyright restrictions"""
    
    try:
        display_copyright_warning()
        
        response = input("\n📝 Type 'I ACKNOWLEDGE' to confirm you understand the restrictions: ")
        
        if response.upper().strip() != "I ACKNOWLEDGE":
            print("\n❌ Authorization denied. Exiting...")
            sys.exit(1)
            
        print("\n✅ Authorization confirmed. Code viewing permitted.")
        return True
        
    except KeyboardInterrupt:
        print("\n\n❌ Process interrupted by user. Exiting...")
        sys.exit(1)

if __name__ == "__main__":
    print("🔒 Copyright Protection System Active")
    check_authorization()