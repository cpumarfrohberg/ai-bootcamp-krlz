import json
import os
import sys
from datetime import datetime

from wikiagent.monitoring.db import LLMLog, get_db


def export_logs_to_json(output_file: str = "logs.json") -> None:
    """Export all logs from database to JSON file."""
    print(f"📊 Exporting logs to {output_file}...")

    with get_db() as db:
        if not db:
            print("❌ Error: Could not connect to database")
            print("   Make sure DATABASE_URL is set and the database is running")
            print("   Try: docker-compose up -d")
            sys.exit(1)

        # Get all logs ordered by creation date
        logs = db.query(LLMLog).order_by(LLMLog.created_at.asc()).all()

        if not logs:
            print("⚠️  No logs found in database")
            print("   Make sure you've run at least one query in the Streamlit app")
            # Create empty file with metadata
            output = {
                "exported_at": datetime.now().isoformat(),
                "total_logs": 0,
                "logs": [],
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"   Created empty {output_file} file")
            sys.exit(0)

        # Convert logs to dictionaries
        logs_data = []
        for log in logs:
            log_dict = {
                "id": log.id,
                "created_at": log.created_at.isoformat() if log.created_at else None,
                "agent_name": log.agent_name,
                "provider": log.provider,
                "model": log.model,
                "user_prompt": log.user_prompt,
                "instructions": log.instructions,
                "total_input_tokens": log.total_input_tokens,
                "total_output_tokens": log.total_output_tokens,
                "assistant_answer": log.assistant_answer,
                "input_cost": log.input_cost,
                "output_cost": log.output_cost,
                "total_cost": log.total_cost,
                "raw_json": json.loads(log.raw_json) if log.raw_json else None,
            }
            logs_data.append(log_dict)

        # Create output structure
        output = {
            "exported_at": datetime.now().isoformat(),
            "total_logs": len(logs_data),
            "logs": logs_data,
        }

        # Write to JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"✅ Successfully exported {len(logs_data)} log(s) to {output_file}")
        print(f"   File size: {os.path.getsize(output_file) / 1024:.2f} KB")
        print("\n📝 Next steps:")
        print(f"   1. Review the file: cat {output_file}")
        print(
            f"   2. Commit to git: git add {output_file} && git commit -m 'Add logs export'"
        )
        print("   3. Push to GitHub: git push")


if __name__ == "__main__":
    export_logs_to_json("logs.json")
