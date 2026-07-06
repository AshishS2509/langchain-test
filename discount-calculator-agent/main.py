from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool

MODEL = "ollama:lfm2.5-thinking:latest"


@tool
def get_product_details(product: str):
    """Get Product details"""
    dict = {
        "laptop": {
            "name": "MacBook Pro",
            "price": 2000,
            "currency": "USD",
            "Manufacturer": "Apple",
            "category": "Electronics",
        },
        "rice": {
            "name": "Indrayani Basmati",
            "price" : 1,
            "currency": "USD",
            "Manufacturer": "Indrayani",
            "category": "Grocery",
            "unit": "kg"
        }
    }
    return dict.get(product, "Product not found")


@tool
def calculate_discount(price: float, discount_percentage: float):
    """Calculate Discount using price and discount percentage"""
    discount_amount = price * (discount_percentage / 100)
    discounted_price = price - discount_amount
    return discounted_price


@tool
def get_discount_percentage(category: str):
    """Get discount percentage based on category"""
    dict = {"Electronics": "10%", "Grocery": "5%", "Apparel": "15%"}
    return dict.get(category, "0%")


def main():
    tools = [get_product_details, calculate_discount, get_discount_percentage]
    tools_dict = {tool.name: tool for tool in tools}
    model = init_chat_model(model=MODEL, temperature=0)
    tools_model = model.bind_tools(tools)
    chat = [
        SystemMessage(content="""
                        To calculate a discounted price:
                        ONLY USE ONE TOOL AT A TIME.
                        1. Get product details using get_product_details.
                        2. Get the discount using get_discount_percentage with param as product category, which you will get fom products details.
                        3. Calculate the final price using calculate_discount.
                        4. If product caregory is Grocery, then you need to multiply the price with the quantity of product to get the final price.
                        Use the provided tools.
                        Never calculate yourself.
                        Never assume discount.
                        Always inspect tool parameters.
                        """),
        HumanMessage(content="What is the discounted price of 5kg of rice?"),
    ]
    for i in range(5):
        response = tools_model.invoke(chat)
        response_tool_calls = response.tool_calls
        if not response_tool_calls:
            print("Chat:", chat)
            print("Response:", response)
            break
        tool_call = response_tool_calls[0]
        tool = tools_dict.get(tool_call["name"])
        if tool is None:
            chat.append(HumanMessage(content=f"Tool {tool_call.name} not found."))
            continue
        tool_response = tool.invoke(tool_call.get("args",{}))
        chat.append(response)
        chat.append(ToolMessage(content=tool_response, tool_call_id=tool_call.get("id")))

if __name__ == "__main__":
    main()
