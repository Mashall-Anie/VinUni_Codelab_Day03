"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""
import os
import re
import json
from typing import List, Dict, Any, Optional

from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""


class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> str:
        # TODO (Milestone 1): Trả về câu trả lời tĩnh hoặc gọi LLM 1 lượt (không dùng tool)
       
        if self.api_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-3.6-flash")
                response = model.generate_content(
                    "Bạn là chatbot tư vấn du lịch. Hãy trả lời câu hỏi sau, "
                    "KHÔNG dùng tool hay internet, chỉ dựa trên kiến thức sẵn có "
                    f"(có thể không chính xác/bịa ra):\n\n{user_input}"
                )
                return f"[Chatbot Baseline] {response.text.strip()}"
            except Exception as e:
                return f"[Chatbot Baseline] Lỗi khi gọi LLM ({e}). Không có tool nên không thể tra cứu dữ liệu thật."

        return (
            "[Chatbot Baseline] Xin lỗi, tôi không có quyền truy cập vào dữ liệu "
            "chuyến bay hay thời tiết thời gian thực, nên không thể trả lời chính "
            f"xác câu hỏi: \"{user_input}\"."
        )


class ReActAgent:
    """Production-grade ReAct Agent with Tool Registry and Safeguards"""

    def __init__(self, max_iterations: int = 5, api_key: str = None):
        self.max_iterations = max_iterations
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.trace: List[Dict[str, Any]] = []

    
    def _tools_description(self) -> str:
        lines = []
        for tool in TOOL_DEFINITIONS:
            lines.append(f"- {tool['name']}({tool['parameters']}): {tool['description']}")
        return "\n".join(lines)

    def _call_llm(self, conversation: str) -> str:
        """
        Gọi LLM thật (nếu có api_key) để sinh ra bước Thought/Action/Final Answer
        tiếp theo. Nếu không có api_key, dùng một "LLM giả lập" đơn giản (rule-based)
        để bài lab vẫn chạy được offline và có thể kiểm thử tự động.
        """
        if self.api_key:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel("gemini-3.6-flash")
            full_prompt = SYSTEM_PROMPT.format(tools=self._tools_description()) + "\n\n" + conversation
            response = model.generate_content(full_prompt)
            return response.text.strip()

        return self._fake_llm_step(conversation)

    def _fake_llm_step(self, conversation: str) -> str:
        """LLM giả lập dùng rule đơn giản, chỉ để demo/test không cần API key."""
        has_flight_obs = "get_flight_info" in conversation and "Observation" in conversation
        has_weather_obs = "get_weather_forecast" in conversation and "Observation" in conversation

        origin, destination = self._extract_route(conversation)

        if not has_flight_obs:
            return (
                "Thought: Tôi cần tìm chuyến bay phù hợp trước.\n"
                'Action: {"name": "get_flight_info", "args": '
                f'{{"origin": "{origin}", "destination": "{destination}", "max_price": 2000000}}}}'
            )

        if not has_weather_obs:
            return (
                "Thought: Tôi đã có thông tin chuyến bay, giờ cần kiểm tra thời tiết điểm đến.\n"
                'Action: {"name": "get_weather_forecast", "args": '
                f'{{"city_code": "{destination}"}}}}'
            )

        return (
            "Thought: Tôi đã có đủ thông tin chuyến bay và thời tiết để trả lời khách hàng.\n"
            "Final Answer: Dựa trên dữ liệu tra cứu được, đây là chuyến bay phù hợp và "
            "gợi ý trang phục theo thời tiết điểm đến (xem chi tiết trong trace)."
        )

    @staticmethod
    def _extract_route(text: str) -> (str, str):
        codes = re.findall(r"\b([A-Z]{3})\b", text.upper())
        known = {"HAN", "SGN", "DAD"}
        found = [c for c in codes if c in known]
        origin = found[0] if len(found) >= 1 else "HAN"
        destination = found[1] if len(found) >= 2 else "SGN"
        return origin, destination

    
    def run(self, user_input: str) -> str:
        # TODO 1: Khởi tạo mảng lưu lịch sử conversation / traces
        self.trace = []
        conversation = f"User: {user_input}\n"
        self.trace.append({"step": "init", "user_input": user_input})

        iteration = 0
        consecutive_tool_errors = 0

        # TODO 2: Thiết lập vòng lặp while iteration < self.max_iterations
        while iteration < self.max_iterations:
            iteration += 1

            # TODO 3: Phân tích Thought / Action từ Agent
            llm_output = self._call_llm(conversation)
            conversation += llm_output + "\n"

            thought = self._extract_field(llm_output, "Thought")
            final_answer = self._extract_field(llm_output, "Final Answer")
            action_raw = self._extract_field(llm_output, "Action")

            step_record = {
                "step": iteration,
                "thought": thought,
                "raw_output": llm_output,
            }

            if final_answer and not action_raw:
                step_record["final_answer"] = final_answer
                self.trace.append(step_record)
                return final_answer

            # TODO 4: Thực thi Tool trong TOOL_MAP nếu có Action
            if action_raw:
                try:
                    action = json.loads(action_raw)
                    tool_name = action.get("name", "")
                    tool_args = action.get("args", {})
                except (json.JSONDecodeError, AttributeError):
                    observation = "Observation: Invalid JSON format"
                    step_record["action"] = action_raw
                    step_record["observation"] = observation
                    self.trace.append(step_record)
                    conversation += observation + "\n"
                    consecutive_tool_errors += 1
                    if consecutive_tool_errors >= 2:
                        return (
                            "Xin lỗi, tôi gặp lỗi liên tục khi xử lý yêu cầu và không thể "
                            "hoàn thành. Vui lòng thử lại hoặc liên hệ hỗ trợ."
                        )
                    continue

                normalized_tool_name = tool_name.strip().lower()
                tool_fn = TOOL_MAP.get(normalized_tool_name)

                if tool_fn is None:
                    observation = f"Observation: Tool '{tool_name}' không tồn tại trong TOOL_MAP"
                    result = {"error": f"Unknown tool: {tool_name}"}
                else:
                    try:
                        result = tool_fn(**tool_args)
                    except TypeError as e:
                        result = {"error": f"Sai tham số khi gọi tool '{tool_name}': {e}"}
                    observation = f"Observation: {json.dumps(result, ensure_ascii=False)}"

                # TODO 5: Ghi lại Observation và lặp lại cho tới khi ra Final Answer
                step_record["action"] = {"name": tool_name, "args": tool_args}
                step_record["observation"] = result
                self.trace.append(step_record)
                conversation += observation + "\n"

                is_error = isinstance(result, dict) and "error" in result
                if is_error:
                    consecutive_tool_errors += 1
                else:
                    consecutive_tool_errors = 0

                if consecutive_tool_errors >= 2:
                    return (
                        "Xin lỗi, tôi không thể lấy được thông tin bạn cần do lỗi lặp lại "
                        "nhiều lần từ hệ thống tra cứu. Vui lòng thử lại sau."
                    )
                continue

            step_record["observation"] = "Không nhận diện được Action hoặc Final Answer hợp lệ."
            self.trace.append(step_record)
            break

        return {
            "status": "max_iterations_reached",
            "answer": "Không thể hoàn thành trong số bước tối đa.",
            "trace": self.trace,
        }

    @staticmethod
    def _extract_field(text: str, field_name: str) -> Optional[str]:
        """Trích xuất nội dung sau 'FieldName:' cho tới hết dòng (hoặc tới field kế tiếp)."""
        pattern = rf"{field_name}:\s*(.*)"
        match = re.search(pattern, text)
        if not match:
            return None
        value = match.group(1).strip()
        return value if value else None


def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()