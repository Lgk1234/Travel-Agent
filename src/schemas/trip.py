"""行程规划相关的数据模型。

在基础行程模型上新增 people_count（人数）与 budget（总预算）字段，
以匹配「填写人数、地点、预算、日期」的规划页需求。
"""
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class TripRequest(BaseModel):
    """行程规划请求。"""

    city: str = Field(..., description="目的地城市")
    start_date: str = Field(..., description="出发日期 YYYY-MM-DD")
    end_date: str = Field(..., description="返程日期 YYYY-MM-DD")
    travel_days: int = Field(default=3, description="出行天数")
    people_count: int = Field(default=1, description="出行人数")
    budget: Optional[int] = Field(default=None, description="总预算（元），可留空")
    transportation: str = Field(default="公共交通", description="交通方式")
    accommodation: str = Field(default="经济型酒店", description="住宿类型")
    preferences: List[str] = Field(default_factory=list, description="偏好标签")
    free_text_input: str = Field(default="", description="自由文本补充需求")


class Attraction(BaseModel):
    """景点。"""

    name: str = Field(..., description="景点名称")
    description: str = Field(default="", description="简介")
    ticket_price: int = Field(default=0, description="门票价格（元）")
    recommended_duration: str = Field(default="", description="建议游玩时长")


class Meal(BaseModel):
    """餐饮。"""

    type: str = Field(..., description="餐别：早餐/午餐/晚餐")
    restaurant: str = Field(default="", description="推荐餐厅")
    recommended_dishes: str = Field(default="", description="推荐菜品")
    estimated_cost: int = Field(default=0, description="预估费用（元）")


class Hotel(BaseModel):
    """酒店。"""

    name: str = Field(..., description="酒店名称")
    price_per_night: int = Field(default=0, description="每晚参考价格（元）")
    rating: str = Field(default="", description="评分")
    address: str = Field(default="", description="地址")
    description: str = Field(default="", description="简介")


class RouteSegment(BaseModel):
    """每日相邻景点/酒店之间的接驳段（由 route_node 基于高德路径规划生成）。"""

    start: str = Field(default="", description="出发地")
    end: str = Field(default="", description="目的地")
    mode: str = Field(default="", description="交通方式，如 地铁/步行/驾车")
    duration: str = Field(default="", description="预计时长，如 约30分钟")
    note: str = Field(default="", description="补充说明")


class DayPlan(BaseModel):
    """每日行程。"""

    date: str = Field(..., description="日期 YYYY-MM-DD")
    day_index: int = Field(..., description="第几天（从 0 开始）")
    description: str = Field(default="", description="当日行程概述")
    transportation: str = Field(default="", description="交通方式")
    accommodation: str = Field(default="", description="住宿类型")
    hotel: Optional[Hotel] = Field(default=None, description="推荐酒店")
    meals: List[Meal] = Field(default_factory=list, description="餐饮列表")
    attractions: List[Attraction] = Field(default_factory=list, description="景点列表")
    route: List[RouteSegment] = Field(
        default_factory=list, description="当日接驳路线（景点/酒店间）"
    )


class WeatherInfo(BaseModel):
    """天气信息。"""

    date: str = Field(..., description="日期 YYYY-MM-DD")
    day_weather: str = Field(default="", description="白天天气")
    night_weather: str = Field(default="", description="夜间天气")
    day_temp: Union[int, str] = Field(default=0, description="白天温度")
    night_temp: Union[int, str] = Field(default=0, description="夜间温度")
    wind_direction: str = Field(default="", description="风向")
    wind_power: str = Field(default="", description="风力")

    @field_validator("day_temp", "night_temp", mode="before")
    @classmethod
    def parse_temp(cls, v):
        if isinstance(v, str):
            v = v.replace("°C", "").replace("℃", "").replace("°", "").strip()
            try:
                return int(v)
            except ValueError:
                return 0
        return v


class Budget(BaseModel):
    """预算信息。"""

    total_attractions: int = Field(default=0, description="景点门票总费用")
    total_hotels: int = Field(default=0, description="酒店总费用")
    total_meals: int = Field(default=0, description="餐饮总费用")
    total_transportation: int = Field(default=0, description="交通总费用")
    total: int = Field(default=0, description="总费用")


class TripPlan(BaseModel):
    """完整旅行计划。"""

    city: str
    start_date: str
    end_date: str
    days: List[DayPlan] = Field(..., description="每日行程")
    weather_info: List[WeatherInfo] = Field(default_factory=list, description="天气信息")
    overall_suggestions: str = Field(default="", description="总体建议")
    budget: Optional[Budget] = Field(default=None, description="预算信息")


class TripPlanResponse(BaseModel):
    """行程规划响应。"""

    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    data: Optional[TripPlan] = Field(default=None, description="旅行计划数据")


class TripReviseRequest(BaseModel):
    """行程修订（Human-in-the-loop 调整）请求。"""

    request: TripRequest = Field(..., description="原始行程请求（用于重新查询工具）")
    plan: TripPlan = Field(..., description="当前行程（作为修订基线）")
    instruction: str = Field(..., description="用户修改意见，如「第二天安排轻松点」「酒店换到离故宫近的」")
