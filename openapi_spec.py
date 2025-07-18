OPENAPI_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "AsesorMatch API", "version": "1.0.0"},
    "paths": {
        "/login": {
            "post": {
                "summary": "Obtener token JWT",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {"userId": {"type": "integer"}},
                                "required": ["userId"],
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Token generado",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"token": {"type": "string"}},
                                }
                            }
                        },
                    },
                    "400": {"description": "Solicitud incorrecta"},
                },
            }
        },
        "/match/calculate": {
            "post": {
                "summary": "Obtener recomendaciones de asesores",
                "security": [{"bearerAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {"studentId": {"type": "integer"}},
                                "required": ["studentId"],
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Lista de recomendaciones",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "advisorId": {"type": "integer"},
                                            "name": {"type": "string"},
                                            "score": {"type": "number"},
                                        },
                                    },
                                }
                            }
                        },
                    },
                    "400": {"description": "Solicitud incorrecta"},
                    "401": {"description": "Token inválido"},
                    "404": {"description": "No encontrado"},
                },
            }
        },
    },
    "components": {
        "securitySchemes": {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
            }
        }
    },
}
