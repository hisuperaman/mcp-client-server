import { Client } from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";
import { GoogleGenAI, type FunctionDeclaration, type Content, } from "@google/genai";
import readline from "readline/promises";
import dotenv from "dotenv";

dotenv.config();

const GEMINI_API_KEY = process.env.GEMINI_API_KEY || '';
const LLM = process.env.LLM || '';

if (!GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY is not set");
}

if (!LLM) {
    throw new Error("LLM is not set");
}

class MCPClient {
    private mcp: Client;
    private gemini: GoogleGenAI;
    private transport: StdioClientTransport | null = null;
    private tools: FunctionDeclaration[] = [];

    constructor() {
        this.gemini = new GoogleGenAI({
            apiKey: GEMINI_API_KEY,
        });

        this.mcp = new Client({
            name: "mcp-client-cli",
            version: "1.0.0",
        });
    }

    async connectToServer(serverScriptPath: string) {
        try {
            const isJs = serverScriptPath.endsWith(".js");
            const isPy = serverScriptPath.endsWith(".py");
            if (!isJs && !isPy) {
                throw new Error("Server script must be a .js or .py file");
            }
            const command = isPy
                ? process.platform === "win32"
                    ? "python"
                    : "python3"
                : process.execPath;

            this.transport = new StdioClientTransport({
                command,
                args: [serverScriptPath],
            });
            await this.mcp.connect(this.transport);

            const toolsResult = await this.mcp.listTools();
            this.tools = toolsResult.tools.map((tool) => {
                return {
                    name: tool.name,
                    description: tool.description,
                    parametersJsonSchema: tool.inputSchema,
                };
            });
            console.log(
                "Connected to server with tools:",
                this.tools.map(({ name }) => name)
            );
        } catch (e) {
            console.log("Failed to connect to MCP server: ", e);
            throw e;
        }
    }

    async processQuery(query: string) {
        const contents: Content[] = [
            {
                role: "user",
                parts: [{ text: query }],
            },
        ];

        const finalText: string[] = [];

        const response = await this.gemini.models.generateContent({
            model: LLM,
            contents,
            config: {
                tools: [
                    {
                        functionDeclarations: this.tools,
                    },
                ],
                maxOutputTokens: 1000,
            },
        });

        for (const part of response.candidates?.[0]?.content?.parts ?? []) {
            if (part.text) {
                finalText.push(part.text);
            }

            if (part.functionCall) {
                const toolName = part.functionCall.name;

                // Gemini's type allows name to be undefined.
                if (!toolName) {
                    continue;
                }

                const toolArgs = part.functionCall.args ?? {};

                finalText.push(
                    `[Calling tool ${toolName} with args ${JSON.stringify(toolArgs)}]`
                );

                const result = await this.mcp.callTool({
                    name: toolName,
                    arguments: toolArgs,
                });

                const toolResult = result.content
                    .filter((block) => block.type === "text")
                    .map((block) => block.text)
                    .join("\n");

                // Add Gemini's function call to the conversation.
                contents.push({
                    role: "model",
                    parts: [
                        {
                            functionCall: part.functionCall,
                        },
                    ],
                });

                // Add the MCP tool result.
                contents.push({
                    role: "user",
                    parts: [
                        {
                            functionResponse: {
                                name: toolName,
                                response: {
                                    result: toolResult,
                                },
                            },
                        },
                    ],
                });

                const finalResponse = await this.gemini.models.generateContent({
                    model: "gemini-2.5-flash",
                    contents,
                    config: {
                        tools: [
                            {
                                functionDeclarations: this.tools,
                            },
                        ],
                        maxOutputTokens: 1000,
                    },
                });

                for (
                    const finalPart of
                    finalResponse.candidates?.[0]?.content?.parts ?? []
                ) {
                    if (finalPart.text) {
                        finalText.push(finalPart.text);
                    }
                }
            }
        }

        return finalText.join("\n");
    }

    async chatLoop() {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
        });

        try {
            console.log("\nMCP Client Started!");
            console.log("Type your queries or 'quit' to exit.");

            while (true) {
                const message = await rl.question("\nQuery: ");
                if (message.toLowerCase() === "quit") {
                    break;
                }
                const response = await this.processQuery(message);
                console.log("\n" + response);
            }
        } finally {
            rl.close();
        }
    }

    async cleanup() {
        await this.mcp.close();
    }
}

async function main() {
    if (process.argv.length < 3) {
        console.log("Usage: node index.ts <path_to_server_script>");
        return;
    }
    const mcpClient = new MCPClient();
    try {
        await mcpClient.connectToServer(process.argv[2]);
        await mcpClient.chatLoop();
    } catch (e) {
        console.error("Error:", e);
        await mcpClient.cleanup();
        process.exit(1);
    } finally {
        await mcpClient.cleanup();
        process.exit(0);
    }
}

main();